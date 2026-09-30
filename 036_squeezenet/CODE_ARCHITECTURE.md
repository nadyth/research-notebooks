# Code Architecture — SqueezeNet Notebook

## Overview

The notebook implements the SqueezeNet Fire module from scratch in PyTorch, builds a compact SqueezeNet adapted for CIFAR-10, and compares its accuracy and parameter count against a standard baseline CNN.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Imports: `torch`, `torch.nn`, `torch.optim`, `torchvision` (datasets, transforms), `numpy`, `matplotlib`.
- Sets device to CUDA if available.
- Installs no extra packages beyond PyTorch and torchvision (both pip-installable on Kaggle).

### 2. Data Loading (CIFAR-10)
- Loads CIFAR-10 with standard transforms (normalize, optionally augment with RandomCrop + HorizontalFlip).
- Creates train/test DataLoaders with batch_size=128.
- CIFAR-10: 50K train / 10K test, 10 classes, 32×32 RGB images.

### 3. Fire Module Implementation
- **`FireModule(nn.Module)`**: the core building block.
  - `squeeze`: 1×1 conv → ReLU
  - `expand_1x1`: 1×1 conv → ReLU
  - `expand_3x3`: 3×3 conv (padding=1) → ReLU
  - Forward: squeeze(x) → split into expand_1x1 and expand_3x3 → concatenate along channel dim.
- Hyperparams: `inplanes`, `squeeze_planes`, `expand1x1_planes`, `expand3x3_planes`.
- Design rule enforced: `squeeze_planes < expand1x1_planes + expand3x3_planes`.

### 4. SqueezeNet Model
- **`SqueezeNet(nn.Module)`**: adapted for CIFAR-10 (32×32 input, 10 classes).
  - `conv1`: 3×3, 64 filters, stride 1 (smaller than original 7×7/stride 2 since CIFAR images are tiny).
  - `maxpool1`: 3×3, stride 2.
  - `fire2`–`fire5`: increasing filter counts.
  - `maxpool2`: after fire4.
  - `fire6`–`fire8`: with bypass connections.
  - `maxpool3`: after fire8.
  - `fire9`: with bypass.
  - `classifier`: 1×1 conv, 10 filters → global avg pool.
  - Bypass: identity shortcut when input/output channels match, else 1×1 conv projection.
- Forward pass applies ReLU after each Fire module output.

### 5. Baseline CNN (Comparison)
- **`BaselineCNN(nn.Module)`**: a standard 3-conv-layer CNN.
  - conv1: 3×3, 64 → conv2: 3×3, 128 → conv3: 3×3, 256 → FC → FC(10).
  - Uses ~2-3× more parameters than SqueezeNet for the same depth.

### 6. Training Loop
- Standard training loop with CrossEntropyLoss and Adam optimizer (lr=0.001).
- Trains for a small number of epochs (10-15) — enough to show the trend on CIFAR-10.
- Records train/test accuracy per epoch.

### 7. Parameter Count Comparison
- Counts total parameters for both models using `sum(p.numel() for p in model.parameters())`.
- Prints a comparison table: model name, total params, test accuracy.

### 8. Visualization
- Plots training curves (loss and accuracy) for both models.
- Bar chart comparing parameter counts.
- Prints final summary table.

## Data Flow / Shapes (CIFAR-10, batch=128)

| Layer | Input Shape | Output Shape |
|-------|------------|-------------|
| Input | (128, 3, 32, 32) | (128, 3, 32, 32) |
| conv1 (3×3, 64, s1) | (128, 3, 32, 32) | (128, 64, 32, 32) |
| maxpool1 (s2) | (128, 64, 32, 32) | (128, 64, 16, 16) |
| fire2 (s=16, e=64) | (128, 64, 16, 16) | (128, 128, 16, 16) |
| fire3 (s=16, e=64) | (128, 128, 16, 16) | (128, 128, 16, 16) |
| fire4 (s=32, e=128) | (128, 128, 16, 16) | (128, 256, 16, 16) |
| maxpool2 (s2) | (128, 256, 16, 16) | (128, 256, 8, 8) |
| fire5 (s=32, e=128) | (128, 256, 8, 8) | (128, 256, 8, 8) |
| fire6 (s=48, e=192) | (128, 256, 8, 8) | (128, 384, 8, 8) |
| fire7 (s=48, e=192) | (128, 384, 8, 8) | (128, 384, 8, 8) |
| fire8 (s=64, e=256) | (128, 384, 8, 8) | (128, 512, 8, 8) |
| maxpool3 (s2) | (128, 512, 8, 8) | (128, 512, 4, 4) |
| fire9 (s=64, e=256) | (128, 512, 4, 4) | (128, 512, 4, 4) |
| classifier (1×1, 10) | (128, 512, 4, 4) | (128, 10, 4, 4) |
| global avg pool | (128, 10, 4, 4) | (128, 10) |

## Deliberate Simplifications vs Full Paper

1. **CIFAR-10 instead of ImageNet**: the paper uses ImageNet (1.2M images, 1000 classes). CIFAR-10 is used for faster training and Kaggle GPU feasibility.
2. **Smaller filter counts**: the original SqueezeNet uses up to 512 filters; we use smaller counts to keep training fast on CIFAR-10's 32×32 images.
3. **Fewer Fire modules**: original has fire2–fire9 (8 modules); we keep the same count but with reduced filter sizes.
4. **No model compression**: the paper's key result includes Deep Compression to <0.5MB. We focus on architecture-only parameter reduction (the first contribution).
5. **Shorter training**: 10-15 epochs instead of the full training schedule used for ImageNet convergence.
6. **conv1 adapted**: 3×3 stride 1 instead of 7×7 stride 2, because CIFAR-10 images are 32×32 (vs ImageNet 224×224).
