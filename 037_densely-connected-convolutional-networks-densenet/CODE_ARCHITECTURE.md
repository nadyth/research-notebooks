# Code Architecture — DenseNet (Densely Connected Convolutional Networks)

## Notebook Overview

The notebook implements a small DenseNet from scratch using PyTorch, trains it on CIFAR-10, and visualizes how feature-map channels grow across a dense block.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `torch`, `torchvision`, `matplotlib` (all pip-installable).
- Imports: `torch`, `torch.nn`, `torch.optim`, `torchvision` for CIFAR-10, `matplotlib` for visualization.

### 2. Dense Layer (Single Unit)
**Class: `DenseLayer(nn.Module)`**
- Input: concatenation of all preceding feature maps → shape `(B, C_in, H, W)`
- Pre-activation: `BatchNorm2d → ReLU → Conv2d(1×1, bottleneck)` → `BatchNorm2d → ReLU → Conv2d(3×3, pad=1)`
- The 1×1 bottleneck conv reduces channels to `4*k` (bottleneck width).
- The 3×3 conv outputs exactly `k` feature maps (growth rate).
- Output: `(B, k, H, W)` — spatial dimensions preserved.

### 3. Dense Block
**Class: `DenseBlock(nn.Module)`**
- Contains `num_layers` DenseLayer modules in a `ModuleList`.
- Forward: iteratively concatenate each layer's output with the growing feature list.
- After layer i: `C_in = C_initial + i * k` → channels grow linearly.
- Data flow: `[x₀] → [x₀, x₁] → [x₀, x₁, x₂] → ... → [x₀, ..., x_L]`
- Final output channels: `C_initial + num_layers * k`

### 4. Transition Layer
**Class: `TransitionLayer(nn.Module)`**
- `BatchNorm2d → ReLU → Conv2d(1×1, compression)` → `AvgPool2d(2, stride=2)`
- Reduces channels to `θ * C_in` (compression factor θ=0.5 for DenseNet-BC).
- Halves spatial dimensions: `(H, W) → (H/2, W/2)`.

### 5. Full DenseNet
**Class: `DenseNet(nn.Module)`**
- Initial conv: `Conv2d(3, 2*k, 3, pad=1) → BatchNorm2d → ReLU` → `2*k` channels at input resolution.
- 3 Dense blocks, each with `num_layers_per_block` layers, interleaved with 2 Transition layers.
- Global average pooling → Linear classifier → 10 output classes (CIFAR-10).
- Architecture: `[InitialConv] → [DenseBlock1] → [Transition1] → [DenseBlock2] → [Transition2] → [DenseBlock3] → [GAP] → [FC]`

### 6. Data Loading
- CIFAR-10 via `torchvision.datasets.CIFAR10`.
- Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, ToTensor, Normalize.
- Batch size: 128. Train/test split from torchvision.

### 7. Training Loop
- Optimizer: SGD with momentum=0.9, weight_decay=1e-4.
- Learning rate scheduler: MultiStepLR (decay at 50% and 75% of epochs).
- Loss: CrossEntropyLoss.
- Epochs: 20 (kept small for time budget; paper uses 300).
- Prints train loss, train acc, test acc per epoch.

### 8. Feature-Map Channel Growth Visualization
- Extracts intermediate feature maps from a DenseBlock.
- For each layer within the block, records the cumulative channel count: `C_initial + (i+1) * k`.
- Plots a bar chart showing channel growth, and a few sample feature maps.
- Demonstrates the key DenseNet property: feature reuse via concatenation grows the representation linearly while each layer only adds k new channels.

## Key Shapes (with growth_rate k=12, 4 layers per block, input 32×32)

| Stage | Channels | Spatial |
|-------|----------|---------|
| Input | 3 | 32×32 |
| Initial conv | 24 (2k) | 32×32 |
| DenseBlock1 (4 layers) | 24 + 4×12 = 72 | 32×32 |
| Transition1 | 36 (0.5×72) | 16×16 |
| DenseBlock2 (4 layers) | 36 + 4×12 = 84 | 16×16 |
| Transition2 | 42 (0.5×84) | 8×8 |
| DenseBlock3 (4 layers) | 42 + 4×12 = 90 | 8×8 |
| GAP | 90 | 1×1 |
| FC | 10 | — |

## Deliberate Simplifications vs. Full Paper

1. **Fewer layers:** 4 layers per block (12 total) vs. paper's 16–40 per block (DenseNet-40, -121, etc.). Keeps training under the Kaggle 30-min limit.
2. **Fewer epochs:** 20 epochs vs. paper's 300. Test accuracy will be ~80–85% instead of 95%+.
3. **No bottleneck in initial config:** The 1×1 bottleneck is included (DenseNet-BC style), but compression θ=0.5 is applied.
4. **CIFAR-10 only:** Paper evaluates on CIFAR-10, CIFAR-100, SVHN, and ImageNet. We use CIFAR-10 for simplicity.
5. **No data parallelism / single GPU:** Paper uses multiple GPUs for ImageNet. We use a single device.
6. **Growth rate k=12:** Matches the paper's smallest setting; keeps parameter count low.
