# Code Architecture — EfficientNet Compound Scaling

## Overview

The notebook implements the core ideas of EfficientNet from scratch:
1. **MBConv block** (Mobile Inverted Bottleneck with SE)
2. **A small baseline CNN** (EfficientNet-B0-inspired, simplified)
3. **Compound scaling** applied to the baseline
4. **Training on CIFAR-10** with several scaled variants
5. **Plotting accuracy vs. FLOPs**

## Notebook Breakdown

### Section 1: Setup & Imports
- PyTorch, torchvision (CIFAR-10), matplotlib
- Device selection (GPU if available)

### Section 2: MBConv Block
**Key class: `MBConvBlock(nn.Module)`**

Implements the Mobile Inverted Bottleneck:
1. **Expansion:** 1×1 conv to expand channels by expansion ratio (default 6)
2. **Depthwise conv:** k×k depthwise (groups = in_channels) with stride
3. **Squeeze-and-Excitation (SE):** global avg pool → FC → ReLU → FC → sigmoid → channel scale
4. **Projection:** 1×1 conv to project back to output channels
5. **Residual:** skip connection if input/output channels match and stride=1

**Data flow:**
- Input: [B, C_in, H, W]
- After expansion: [B, C_in * expansion, H, W]
- After depthwise: [B, C_in * expansion, H/s, W/s]
- After SE: [B, C_in * expansion, H/s, W/s] (channel-reweighted)
- After projection: [B, C_out, H/s, W/s]
- After residual: [B, C_out, H/s, W/s]

### Section 3: EfficientNet-B0 Baseline
**Key class: `EfficientNet(nn.Module)`**

Simplified baseline inspired by B0:
- Stem: 3×3 conv, stride 1, 32 channels (simplified for CIFAR-10's 32×32 images)
- 7 stages of MBConv blocks with varying channels, repeat counts, and strides
- Head: 1×1 conv to 1280 channels → adaptive avg pool → FC → 10 classes
- Swish activation throughout

**Architecture table (baseline, φ=0):**
| Stage | Blocks | Channels | Stride | Expansion |
|-------|--------|----------|--------|-----------|
| 1     | 1      | 16       | 1      | 1         |
| 2     | 2      | 24       | 2      | 6         |
| 3     | 2      | 40       | 2      | 6         |
| 4     | 3      | 80       | 2      | 6         |
| 5     | 3      | 112      | 1      | 6         |
| 6     | 4      | 192      | 2      | 6         |
| 7     | 1      | 320      | 1      | 6         |

(Adapted for CIFAR-10's smaller resolution; strides reduced to prevent excessive downsampling.)

### Section 4: Compound Scaling Function
**Key function: `scale_efficientnet(model, phi, alpha=1.2, beta=1.1, gamma=1.15)`**

- Computes: depth_scale = alpha^phi, width_scale = beta^phi, resolution_scale = gamma^phi
- Scales number of blocks per stage: round(repeats * depth_scale)
- Scales channels per stage: round(channels * width_scale)
- Returns a new EfficientNet with scaled configuration
- Resolution scaling handled by adjusting input image size (interpolation)

### Section 5: Data Loading
- CIFAR-10 with standard transforms
- For resolution scaling, images are resized to target resolution
- Data augmentation: random crop, horizontal flip, normalize

### Section 6: Training Loop
- CrossEntropyLoss + AdamW optimizer
- Cosine annealing LR scheduler
- Training for ~10 epochs (kept short for Kaggle time limits)
- Tracks training loss and validation accuracy

### Section 7: Running Scaled Variants
- φ = 0, 1, 2, 3 (B0 through B3-equivalent)
- For each φ: build model, train, record accuracy and estimated FLOPs
- FLOPs estimated via `thop` library (profile function)

### Section 8: Plotting Accuracy vs FLOPs
- Scatter plot: x=FLOPs (log scale), y=accuracy
- Annotate each point with φ value
- Show that compound scaling yields better accuracy/FLOP trade-off

## Deliberate Simplifications vs. Full Paper

1. **No neural architecture search** — we use a hand-designed B0 architecture inspired by the paper's table, not the NAS-found one.
2. **CIFAR-10 instead of ImageNet** — 32×32 images, 10 classes, much faster to train.
3. **Reduced epochs** — 10 epochs per variant vs. 350+ in the paper.
4. **Simplified resolution scaling** — CIFAR-10 images are small; resolution scaling uses interpolation up to 64×64 or 96×96 rather than 300×300+.
5. **No compound coefficient grid search** — we use the paper's reported α=1.2, β=1.1, γ=1.15 directly.
6. **No mixed precision training** — the paper uses AMP; we use standard FP32 for simplicity.
7. **No Swish memory-efficient implementation** — standard Swish/SiLU is used.
