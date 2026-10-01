# ResNeXt — Notebook Architecture

## Goal

Implement the ResNeXt cardinality block (grouped-convolution bottleneck) from scratch in PyTorch, build a small ResNeXt and a parameter-matched plain ResNet, train both on CIFAR-10, and compare accuracy and parameter efficiency.

## Section-by-section breakdown

### 1. Setup and data loading
- Imports `torch`, `torchvision`, `matplotlib`, `numpy`.
- Attempts to load CIFAR-10 from local cache; falls back to a synthetic dataset with the same 3×32×32 shape and class-conditional signal if download fails.
- Standard normalization and augmentation: random crop (padding=4), horizontal flip, per-channel mean/std normalization.
- Batch size 128.

### 2. Plain ResNet bottleneck block (baseline)
- Implements the standard ResNet bottleneck block: `1×1 conv → BN → ReLU → 3×3 conv → BN → ReLU → 1×1 conv → BN`, with identity or projection shortcut.
- This is the "plain" block with cardinality C=1 (single path).
- `PlainBottleneckBlock(in_ch, mid_ch, out_ch, stride=1)`.

### 3. ResNeXt cardinality block (core contribution)
- Implements the aggregated residual transformation using **grouped convolutions** (Fig. 3(c) in the paper).
- Structure: `1×1 conv (wide) → BN → ReLU → 3×3 grouped conv (groups=cardinality) → BN → ReLU → 1×1 conv → BN`, with shortcut.
- The 3×3 conv has `groups=C`, splitting the intermediate channels into C groups of `d` channels each. Total intermediate = C × d.
- `ResNeXtBottleneckBlock(in_ch, mid_ch, out_ch, cardinality=32, stride=1)`.
- Key insight: grouped conv is mathematically equivalent to running C independent transformations and summing — but implemented in a single layer.

### 4. Network builders
- `ResNet(blocks_per_stage, channels, block_cls, num_classes=10)`:
  - Initial 3×3 conv (64 channels), BN, ReLU.
  - 4 stages with blocks_per_stage blocks each, channels doubled and spatial halved per stage.
  - Global average pool → FC layer.
- `PlainResNet50()`: 3,4,6,3 blocks with standard bottleneck (cardinality=1). ~25.6M params.
- `ResNeXt50()`: 3,4,6,3 blocks with cardinality=32, d=4. ~25.0M params. Matches ResNet-50 complexity.
- Small variants for CIFAR-10: fewer blocks and channels to keep training feasible on GPU within time budget.

### 5. Parameter count comparison
- Prints parameter counts for both models to verify they are within ~2% of each other (the paper's key constraint).
- Also prints FLOPs estimate using `thop` if available, otherwise manual computation.

### 6. Training loop
- SGD with Nesterov momentum 0.9, weight decay 5e-4.
- Cosine annealing LR schedule from 0.1 to 0.
- Cross-entropy loss.
- Standard `train_epoch`/`evaluate` functions.
- 5 epochs for feasibility (paper used 100+ epochs with longer schedules).

### 7. Comparison: ResNeXt vs Plain ResNet
- Train both models with the same training recipe.
- Plot training loss and test accuracy curves.
- Bar chart comparing final test accuracy and parameter counts.
- Print a summary table.

## Key data flow / shapes (CIFAR-10, small variant)

- Input: `[B, 3, 32, 32]`
- After conv1: `[B, 64, 32, 32]`
- Stage 1 (stride 1): `[B, 256, 32, 32]` (for ResNeXt: 1×1→128, 3×3 grouped(32)→4 per group=128, 1×1→256)
- Stage 2 (stride 2): `[B, 512, 16, 16]`
- Stage 3 (stride 2): `[B, 1024, 8, 8]`
- Global avg pool: `[B, 1024, 1, 1]` → flatten `[B, 1024]`
- FC: `[B, 10]`

## Deliberate simplifications vs full paper

- **CIFAR-10 instead of ImageNet-1K**: 32×32 images, 10 classes vs 224×224, 1000 classes. The paper's full ResNeXt-50 (25M params) is too heavy for quick training; we use a smaller variant.
- **5 epochs instead of 100+**: The paper trained for 120k iterations with 8 GPUs. We use 5 epochs to demonstrate the architecture and show the training dynamic.
- **No COCO detection or ImageNet-5K experiments**: the paper validates on these too; we focus only on classification.
- **Fixed cardinality=32**: the paper sweeps cardinality ∈ {1,2,4,8,32,64}; we use 32 (the recommended value) and compare to cardinality=1 (plain ResNet).
- **No half-precision / multi-GPU training**: the paper uses FP16 and 8 GPUs for speed; we train in FP32 on a single GPU.
