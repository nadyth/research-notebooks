# Code Architecture — Xception Notebook

## Overview

The notebook implements the Xception module (depthwise separable convolutions with residual connections) from scratch in PyTorch, builds a small Xception-style network for CIFAR-10, and compares its accuracy and parameter count against an Inception-style block network of similar depth.

## Section-by-Section Breakdown

### 1. Imports & Setup
- PyTorch, torch.nn, torchvision (CIFAR-10), matplotlib, time
- Device selection (GPU if available)
- Reproducibility seed

### 2. Separable Convolution Block (from scratch)
- **`SeparableConv`** module — the Xception building block:
  - `nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)` — depthwise (spatial filtering per channel, `groups=in_ch`)
  - `nn.BatchNorm2d(in_ch)` — BatchNorm after depthwise (paper specifies all conv layers followed by BN)
  - **No activation between depthwise and pointwise** — this is the key Xception finding
  - `nn.Conv2d(in_ch, out_ch, kernel_size=1)` — pointwise (1×1 conv to combine channels)
  - `nn.BatchNorm2d(out_ch)` — BatchNorm after pointwise
  - `nn.ReLU()` — activation only after the full separable conv block (not between depthwise/pointwise)
- Key design choice: the paper shows that omitting the intermediate non-linearity between depthwise and pointwise operations yields faster convergence and better final performance

### 3. Xception Module (with residual connections)
- **`XceptionModule`** module — wraps SeparableConv blocks with a linear residual skip:
  - Two sequential `SeparableConv` blocks (depthwise separable → BN → ReLU × 2)
  - Residual connection: if input/output channels match, identity skip; if channel count changes, 1×1 conv projection
  - No intermediate activation between the two separable convs (just BN)

### 4. Inception-Style Block (for comparison)
- **`InceptionBlock`** module — a simplified Inception module:
  - 1×1 conv to reduce channels to 3 sub-spaces (each 1/3 of output channels)
  - Three parallel branches: 1×1 conv, 3×3 conv, 5×5 conv (with padding)
  - Concatenate outputs along channel dimension
  - Followed by BatchNorm + ReLU
  - Residual connection with 1×1 projection if needed
- This is the "intermediate point on the spectrum" — channels partitioned into a few segments, not fully decoupled

### 5. Small Xception Architecture
- **`SmallXception`** — scaled-down Xception for CIFAR-10 (32×32 images):
  - Entry flow: standard 3×3 conv (3→32 channels) → 2 XceptionModules (32→64, 64→128, stride-2 downsampling)
  - Middle flow: 3 XceptionModules (128→128, repeated)
  - Exit flow: 2 XceptionModules (128→256, 256→10) → global average pooling → FC
  - Matches the paper's entry/middle/exit structure at a fraction of the scale

### 6. Small Inception-Style Architecture
- **`SmallInceptionNet`** — same depth, same channel progression, same spatial downsampling
  - InceptionBlocks instead of XceptionModules
  - Same FC head and global average pooling
  - This is the apples-to-apples comparison: same parameter budget, different module type

### 7. Parameter Counting
- **`count_parameters(model)`**: sums all trainable parameters
- Manual comparison showing Xception's depthwise separable approach vs Inception's multi-branch approach at equal depth
- Print table: model, total params, params per layer type

### 8. Training on CIFAR-10
- Load CIFAR-10 with standard transforms (normalize, random crop, horizontal flip)
- Train both SmallXception and SmallInceptionNet for 15 epochs
- Same optimizer (SGD with momentum 0.9), same learning rate schedule, same batch size
- Track training loss and test accuracy per epoch

### 9. Results Comparison
- Plot: parameter count bar chart (Xception vs Inception-style)
- Plot: training/test accuracy curves overlaid for both models
- Plot: training loss curves
- Print final accuracy comparison table — demonstrates Xception achieves comparable or better accuracy with similar or fewer parameters

## Key Functions/Classes

| Class/Function | Purpose |
|---|---|
| `SeparableConv` | Depthwise conv (groups=in_ch) → BN → pointwise 1×1 conv → BN → ReLU. No intermediate activation. |
| `XceptionModule` | Two SeparableConv blocks + residual skip (identity or 1×1 projection). |
| `InceptionBlock` | 1×1 reduction → parallel 1×1/3×3/5×5 → concat → BN → ReLU + residual. |
| `SmallXception` | Entry/middle/exit flow with XceptionModules, global avg pool, FC. |
| `SmallInceptionNet` | Same structure with InceptionBlocks for comparison. |
| `count_parameters` | Sum trainable parameters. |
| `train_model` | Training loop with SGD, LR scheduling, loss/accuracy tracking. |
| `plot_comparison` | Accuracy curves, loss curves, parameter count bar charts. |

## Data Flow / Shapes (CIFAR-10, batch=128)

```
Input: [128, 3, 32, 32]
  → Conv 3→32: [128, 32, 32, 32]
  → XceptionModule 32→64, stride 2: [128, 64, 16, 16]
  → XceptionModule 64→128, stride 2: [128, 128, 8, 8]
  → 3× XceptionModule 128→128: [128, 128, 8, 8]
  → XceptionModule 128→256, stride 2: [128, 256, 4, 4]
  → XceptionModule 256→10: [128, 10, 4, 4]
  → Global avg pool: [128, 10]
  → Output: [128, 10]
```

## Deliberate Simplifications vs Full Paper

1. **Scale:** Full Xception has 36 conv layers (14 modules, 8 middle-flow repeats); our SmallXception has ~18 conv layers (7 modules, 3 middle-flow repeats) — scaled for CIFAR-10's 32×32 images and Kaggle GPU time limits.
2. **Dataset:** Full paper uses ImageNet (1.28M images, 1000 classes) and JFT (350M images, 17K classes); we use CIFAR-10 (50K images, 10 classes).
3. **Optimizer:** Paper uses SGD with carefully tuned LR (0.045, decay 0.94/2 epochs) + Polyak averaging; we use SGD with cosine annealing, no Polyak averaging.
4. **No intermediate activation experiment:** The paper separately tests ReLU/ELU between depthwise/pointwise (Figure 10); our notebook implements the "no activation" version (the paper's best-performing configuration) as the default.
5. **Weight decay:** Paper uses 1e-5; we use 5e-4 (standard for CIFAR-10 with small models).
6. **Entry/exit flow:** Simplified channel progressions (32→64→128→256 vs 32→128→256→728→1024→2048→4096 in the full paper).
