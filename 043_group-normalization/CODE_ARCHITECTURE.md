# Code Architecture — Group Normalization Notebook

## Overview

The notebook implements Group Normalization from scratch in both NumPy and PyTorch, plugs it into a small CNN alongside BatchNorm, and trains both under varying batch sizes to demonstrate GN's robustness when batch sizes shrink. All computation is GPU-friendly but also runs on CPU.

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, `matplotlib` for visualization
- Random seed fixing for reproducibility
- Device detection (GPU if available, else CPU)

### 2. GroupNorm from Scratch in NumPy
- **Function `group_norm_numpy(x, num_groups, gamma, beta, eps=1e-5)`**
  - Input: `x` is a 4D array of shape (N, C, H, W)
  - Reshape to (N, G, C//G, H, W) where G = num_groups
  - Compute mean μ and variance σ² over the (C//G, H, W) axes for each (N, G) pair
  - Normalize: x_norm = (x - μ) / sqrt(σ² + eps)
  - Apply per-channel scale γ and shift β
  - Reshape back to (N, C, H, W)
  - Returns normalized output
- **Gradient verification**: numerical gradient check against PyTorch autograd
  - Verify gradients via finite differences on a small random input
  - Assert relative error < 1e-4

### 3. GroupNorm as a PyTorch Module
- **Class `GroupNorm(nn.Module)`**
  - `__init__(num_channels, num_groups, eps=1e-5)`: initializes `self.gamma = nn.Parameter(torch.ones(num_channels))` and `self.beta = nn.Parameter(torch.zeros(num_channels))`
  - `forward(x)`: x has shape (N, C, H, W)
    - Reshape to (N, G, C//G, H, W)
    - Compute mean and variance over dims (2, 3, 4) with keepdim=True
    - Normalize and apply gain/bias
    - Reshape back to (N, C, H, W)
  - Key shapes:
    - Input: (B, C, H, W) → reshape (B, G, C//G, H, W)
    - Mean/var: (B, G, 1, 1, 1)
    - Output: (B, C, H, W)

### 4. BatchNorm2d for Comparison
- **Class `BatchNorm2d(nn.Module)`** — simplified from-scratch version
  - Normalizes over the batch dimension (dim=0) per channel
  - Maintains running mean/var for inference
  - `forward(x)`: during training, use batch statistics and update running stats; during eval, use running stats
  - Key difference from GN: statistics computed across (N, H, W) for each channel

### 5. Normalization Comparison Visualization
- Create a synthetic feature map and show how BN, GN, LayerNorm, and InstanceNorm normalize differently
- Visualize the grouping strategy with a diagram-like output showing which axes each method normalizes over

### 6. Small CNN Architecture
- **Class `ConvBlock(nn.Module)`**
  - Conv2d → Normalization → ReLU
  - Pluggable normalization: BatchNorm, GroupNorm, or None
- **Class `SmallCNN(nn.Module)`**
  - 3 conv blocks with max pooling
  - Channels: 3 → 16 → 32 → 64
  - Global average pooling → linear classifier
  - Forward: (B, 3, 32, 32) → conv blocks → (B, 64, 1, 1) → Linear(64, 10)

### 7. Training Under Varying Batch Sizes
- Dataset: CIFAR-10 (or synthetic if offline)
- Train separate models with BatchNorm and GroupNorm at batch sizes: 32, 8, 4, 2, 1
- Optimizer: SGD with momentum 0.9, lr=0.01
- Loss: CrossEntropyLoss
- Epochs: 5 per configuration (keep runtime short)
- Record final test accuracy for each (norm_type, batch_size) pair
- Key data flow:
  ```
  Input (B, 3, 32, 32) → ConvBlock1 → (B, 16, 16, 16)
    → ConvBlock2 → (B, 32, 8, 8)
    → ConvBlock3 → (B, 64, 4, 4)
    → AdaptiveAvgPool → (B, 64, 1, 1)
    → Linear(64, 10) → logits (B, 10)
  ```

### 8. Visualization & Comparison
- Plot 1: Accuracy vs batch size for BatchNorm vs GroupNorm
- Plot 2: Training loss curves at batch_size=2 (showing BN's instability)
- Plot 3: Feature map statistics (mean/variance) comparison at different batch sizes
- Plot 4: Bar chart of final accuracies across all configurations

### 9. Weight Standardization (Bonus)
- Brief demonstration of Weight Standardization combined with GN
- Shows how WS+GN can match BN performance even at batch size 1

### 10. Summary & Analysis
- Print final accuracies for all configurations
- Discuss: GN maintains stable accuracy as batch size shrinks; BN degrades rapidly below batch size 8
- Show that GN with G=1 is equivalent to LayerNorm and G=C is equivalent to InstanceNorm

## Key Functions/Classes
| Component | Purpose |
|---|---|
| `group_norm_numpy()` | Pure NumPy implementation for gradient verification |
| `GroupNorm(nn.Module)` | Reusable PyTorch GroupNorm module |
| `BatchNorm2d(nn.Module)` | From-scratch BatchNorm for comparison |
| `ConvBlock` | Conv + Norm + ReLU building block |
| `SmallCNN` | 3-block CNN for image classification |
| `train_model()` | Training loop with configurable batch size and norm type |

## Data Flow & Shapes
```
Input: (B, 3, 32, 32)   # CIFAR-10 images
  ↓ ConvBlock1: Conv(3→16, 3x3, pad=1) + Norm + ReLU + MaxPool(2)
  (B, 16, 16, 16)
  ↓ ConvBlock2: Conv(16→32, 3x3, pad=1) + Norm + ReLU + MaxPool(2)
  (B, 32, 8, 8)
  ↓ ConvBlock3: Conv(32→64, 3x3, pad=1) + Norm + ReLU + MaxPool(2)
  (B, 64, 4, 4)
  ↓ AdaptiveAvgPool2d(1)
  (B, 64, 1, 1) → flatten → (B, 64)
  ↓ Linear(64, 10)
  logits: (B, 10)
  ↓ CrossEntropy
  loss: scalar
```

## Deliberate Simplifications vs Full Paper
1. **Small CNN** — 3 conv blocks with 16/32/64 channels (paper uses ResNet-50 with 50 layers)
2. **CIFAR-10** — 32×32 images (paper uses ImageNet at 224×224)
3. **5 epochs** — short training to keep Kaggle runtime under 30 min (paper trains for full schedules)
4. **No detection/segmentation experiments** — paper shows COCO detection and Kinetics video results; we focus on classification accuracy vs batch size
5. **No transfer learning** — paper demonstrates GN transferring from pre-training to fine-tuning; we train from scratch
6. **Fixed group count** — we use G=32 as default (paper also uses 32 for ResNet-50); we briefly explore other G values but don't do a full sweep
7. **No running statistics for GN** — GN doesn't need them by design; we highlight this as an advantage rather than implementing a full eval-mode comparison
