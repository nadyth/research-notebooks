# Code Architecture — MobileNets Notebook

## Overview

The notebook implements depthwise separable convolutions from scratch in PyTorch, builds a small MobileNet-style CNN and a standard-convolution equivalent, trains both on CIFAR-10, and benchmarks parameter count, FLOPs, and inference speed.

## Section-by-Section Breakdown

### 1. Imports & Setup
- PyTorch, torch.nn, torchvision (CIFAR-10), time, functools
- Device selection (GPU if available)
- Reproducibility seed

### 2. Depthwise Separable Convolution (from scratch)
- **`DepthwiseSeparableConv`** module: two sub-layers
  - `nn.Conv2d(in_ch, in_ch, kernel_size, padding, groups=in_ch)` — depthwise (one filter per channel, `groups=in_ch`)
  - `nn.BatchNorm2d(in_ch)` + `nn.ReLU6` — BN+activation after depthwise
  - `nn.Conv2d(in_ch, out_ch, kernel_size=1)` — pointwise (1×1 conv to combine channels)
  - `nn.BatchNorm2d(out_ch)` + `nn.ReLU6` — BN+activation after pointwise
- Key point: `groups=in_ch` in PyTorch is the mechanism for depthwise convolution — each input channel gets its own filter

### 3. Standard Convolution Block (for comparison)
- **`StandardConv`** module: single `nn.Conv2d(in_ch, out_ch, kernel_size, padding)` + BatchNorm + ReLU
- This is the "un-factorized" equivalent — same input/output channels and kernel size

### 4. Small MobileNet Architecture
- **`SmallMobileNet`**: a scaled-down MobileNet for CIFAR-10 (32×32 images)
  - First layer: standard 3×3 conv, stride 1, 3→32 channels (mirrors paper's first full conv layer)
  - 4 depthwise separable blocks with channel progression 32→64→128→256, stride-2 on blocks 2 and 4 for downsampling
  - Global average pooling → FC layer → 10 output classes
  - Width multiplier α applied to all channel counts

### 5. Standard Convolution Equivalent
- **`SmallStandardNet`**: same channel progression and spatial downsampling, but every depthwise separable block replaced by a standard 3×3 convolution
  - Same depth, same channel widths, same pooling/FC head
  - This is the apples-to-apples comparison the paper makes in Table 4

### 6. Parameter & FLOP Counting
- **`count_parameters(model)`**: sums all trainable parameters
- **Manual FLOP estimation**: for each conv layer, compute D_K²·M·N·D_F² (standard) or D_K²·M·D_F² + M·N·D_F² (depthwise separable)
- Print comparison table showing the ~8–9× reduction

### 7. Inference Speed Benchmark
- **`benchmark_inference(model, input_shape, n_runs=100)`**: warm-up + timed runs
- Measures wall-clock time per forward pass for both models
- Reports speedup ratio

### 8. Training on CIFAR-10
- Load CIFAR-10 with standard transforms (normalize, random crop, horizontal flip)
- Train both SmallMobileNet and SmallStandardNet for a modest number of epochs (10–15)
- Same optimizer (Adam), same learning rate schedule, same batch size
- Track training loss and test accuracy per epoch

### 9. Results Comparison
- Plot: parameter count bar chart (MobileNet vs Standard)
- Plot: inference speed bar chart
- Plot: training accuracy curves overlaid
- Print final accuracy comparison table — demonstrates MobileNet achieves comparable accuracy with far fewer parameters

## Key Functions/Classes

| Class/Function | Purpose |
|---|---|
| `DepthwiseSeparableConv` | From-scratch depthwise + pointwise conv block with BN+ReLU |
| `StandardConv` | Equivalent standard conv block for comparison |
| `SmallMobileNet` | Scaled-down MobileNet for CIFAR-10 |
| `SmallStandardNet` | Standard-conv equivalent with same topology |
| `count_parameters()` | Counts trainable parameters |
| `estimate_macs()` | Manually estimates multiply-adds for conv layers |
| `benchmark_inference()` | Times forward passes for speed comparison |
| `train_model()` | Training loop with loss/accuracy tracking |

## Data Flow / Shapes (CIFAR-10, 32×32×3)

```
Input:  [B, 3, 32, 32]
→ Conv 3×3, stride 1:  [B, 32, 32, 32]
→ DW-Sep block 1 (s=1): [B, 64, 32, 32]
→ DW-Sep block 2 (s=2): [B, 128, 16, 16]
→ DW-Sep block 3 (s=1): [B, 256, 16, 16]
→ DW-Sep block 4 (s=2): [B, 256, 8, 8]
→ Global Avg Pool:     [B, 256, 1, 1]
→ Flatten + FC:         [B, 10]
```

## Deliberate Simplifications vs Full Paper

1. **Scale:** Full MobileNet uses 28 layers, 1024 final channels, and 224×224 ImageNet images. We use a 9-layer, 256-channel version on 32×32 CIFAR-10 to keep training under 30 minutes on Kaggle GPU.
2. **Width/resolution multipliers:** We implement the core depthwise separable convolution and demonstrate the parameter/FLOP reduction. We include a width multiplier demonstration but do not sweep all 16 α×ρ combinations from the paper.
3. **Dataset:** CIFAR-10 (10 classes, 32×32) instead of ImageNet (1000 classes, 224×224). The architectural principles are identical.
4. **Training setup:** Adam optimizer instead of RMSprop with async gradient descent (paper used TensorFlow distributed training). No weight decay tuning for depthwise filters.
5. **ReLU6:** We use ReLU6 as in the original MobileNet (useful for quantization), though standard ReLU would also work for this scale.
6. **No distillation, detection, or geolocalization experiments** — those are application case studies in the paper, not core architectural contributions.
