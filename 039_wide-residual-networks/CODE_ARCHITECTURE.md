# Wide Residual Networks — Notebook Architecture

## Goal

The notebook implements a Wide Residual Network (WRN) from scratch in PyTorch, trains it on CIFAR-10 (or a synthetic fallback with the same 3×32×32 shape when CIFAR-10 is unavailable), and compares accuracy and training speed against a deep-narrow ResNet of similar parameter count. It also demonstrates the effect of dropout inside residual blocks.

## Section-by-section breakdown

### 1. Setup and data loading
- Imports `torch`, `torchvision`, `matplotlib`, `numpy`.
- Attempts to load CIFAR-10 from a local `./data/cifar-10-batches-py` directory (no download during execution to avoid hanging on slow connections).
- If CIFAR-10 is not present, falls back to a **synthetic dataset** with the same shape (3×32×32 images, 10 classes) and class-conditional signal so the models can learn. This allows the architecture comparison to run on any CPU without network dependencies.
- Batch size 64, standard normalization and augmentation (random crop + horizontal flip) for real CIFAR-10.

### 2. Basic pre-activation residual block
- Implements the "basic" residual block with **pre-activation** ordering: `BN → ReLU → Conv → BN → ReLU → Conv`, following He et al. (2016) identity-mapping variant.
- The shortcut is identity when in/out channels match; otherwise a 1×1 conv (no bias) projects dimensions.
- Stride of 2 in the first conv of a stage halves spatial resolution.

### 3. Wide residual block
- Same structure as the basic block but the number of output channels is multiplied by the **widening factor k**.
- Optional **dropout** between the two convolutions (the paper's key regularization contribution).
- `WideResidualBlock(in_ch, out_ch, stride, dropout_p)`.

### 4. Network builders
- `WideResNet(depth, widen_factor, dropout_p, num_classes)`:
  - Computes `N = (depth - 4) / 6` blocks per stage (3 stages).
  - Initial conv: 16 channels.
  - Stage 1: 16×k channels, stride 1.
  - Stage 2: 32×k channels, stride 2.
  - Stage 3: 64×k channels, stride 2.
  - Global average pool + FC layer.
- `DeepNarrowResNet(depth, num_classes)`: standard thin ResNet with k=1 for comparison.

### 5. Parameter count check
- WRN-16-2: ~691K parameters (16 layers, width factor 2).
- DeepNarrow-40: ~564K parameters (40 layers, width factor 1).
- Comparable parameter counts with very different depth/width tradeoffs.

### 6. Training loop
- SGD with Nesterov momentum 0.9, weight decay 5e-4.
- Cosine annealing learning rate schedule.
- Cross-entropy loss.
- Reusable `train_epoch`/`evaluate` functions tracking loss and accuracy.
- 3 epochs for CPU feasibility; the paper used 200 epochs.

### 7. Comparison: Wide vs Deep-Narrow
- Train WRN-16-2 (wide, 16 layers, k=2) vs DeepNarrow-40 (thin, 40 layers, k=1).
- Plot training/test accuracy and loss curves on the same axes.
- Bar chart of per-epoch training time.
- Expected: WRN-16-2 trains faster per epoch (~13s vs ~21s on CPU) while achieving comparable accuracy.

### 8. Effect of dropout in residual blocks
- Train WRN-16-2 with and without dropout (p=0.3) and compare.
- Expected: dropout provides regularization, especially for wider networks.

### 9. Summary
- Side-by-side accuracy curves and final test accuracies for all comparisons.

## Key functions / classes

- `BasicPreActBlock(in_ch, out_ch, stride)` — standard thin pre-activation residual block
- `WideResidualBlock(in_ch, out_ch, stride, dropout_p)` — wide block with optional dropout
- `WideResNet(depth, widen_factor, dropout_p, num_classes)` — full WRN builder
- `DeepNarrowResNet(depth, num_classes)` — comparison thin ResNet
- `train_epoch(model, loader, criterion, optimizer, device)` — one epoch of training
- `evaluate(model, loader, criterion, device)` — evaluation on test set
- `count_parameters(model)` — returns total trainable parameter count
- `train_model(model, tr, te, epochs, lr, device, name)` — full training with history tracking

## Data flow and shapes

Input image: `(B, 3, 32, 32)`

Initial conv: `(B, 16, 32, 32)`

Stage 1 (stride 1, k=2): `(B, 32, 32, 32)` × N blocks

Stage 2 (stride 2, k=2): `(B, 64, 16, 16)` × N blocks

Stage 3 (stride 2, k=2): `(B, 128, 8, 8)` × N blocks

Global average pool: `(B, 128)`

FC: `(B, 128)` → `(B, 10)`

For WRN-16-2: N = (16-4)/6 = 2 blocks per stage, total 6 blocks = 12 conv layers + initial conv + FC = ~16 layers.

## Deliberate simplifications vs. full paper

- The paper trains for 200 epochs with cosine annealing on multiple GPUs; we use 3 epochs for CPU feasibility. The relative comparison (wide vs narrow) still holds.
- The paper tests on CIFAR-10, SVHN, COCO, and ImageNet; we use CIFAR-10 or a synthetic fallback.
- We use k=2 (paper recommended k=10) with depth=16 to keep CPU runtime under 10 minutes. The paper's WRN-28-10 would be too slow on CPU.
- The paper's full ablation studies (convolution type, B(3,3) vs B(3,1,3), number of convs per block) are not reproduced; we focus on the width vs depth comparison and dropout effect.
- He initialization and standard PyTorch BN are used instead of custom implementations.
- On Kaggle GPU, the notebook will automatically use real CIFAR-10 if the data directory exists; otherwise the synthetic dataset is used.
