# Code Architecture: SimCLR Notebook

## Overview

The notebook implements SimCLR from scratch on a small scale using CIFAR-10 and a lightweight ResNet encoder. It follows the paper's framework: augmentation → encoder → projection head → NT-Xent loss → linear evaluation.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install/import PyTorch, torchvision, numpy, matplotlib
- Set device (GPU if available), random seeds for reproducibility

### 2. Data Augmentation Pipeline
- **`SimCLRAugment`**: A composition of random crop + resize, color jitter, grayscale, and Gaussian blur
- Two independent augmentations applied to each image to create positive pairs
- Uses torchvision transforms composed into a callable

### 3. Contrastive Dataset Wrapper
- **`ContrastiveDataset`**: Wraps CIFAR-10, returns two augmented views per sample
- For each image: `(view1, view2, label)` — label only used for linear eval

### 4. Base Encoder
- **Small ResNet**: A simplified ResNet-18 (4 residual blocks, ~1M parameters) from torchvision
- Output: 512-dimensional feature vector h

### 5. Projection Head
- **`ProjectionHead`**: 2-layer MLP (512 → 256 → 128) with ReLU activation
- Input: h from encoder, Output: z (128-dim) for contrastive loss
- Discarded after pretraining

### 6. SimCLR Model
- **`SimCLR`**: Combines encoder + projection head
- Forward pass: takes two views, returns (z_i, z_j) projected features

### 7. NT-Xent Loss
- **`nt_xent_loss(z_i, z_j, temperature)`**: 
  - Concatenate all z_i and z_j in batch → 2N × 128 matrix
  - Compute cosine similarity matrix (2N × 2N)
  - Mask out self-similarity
  - For each positive pair (i, j), compute -log(softmax over all negatives)
  - Temperature τ = 0.5

### 8. Pretraining Loop
- Batch size: 256 (or 128 for memory)
- Optimizer: LARS / Adam with cosine learning rate schedule
- Epochs: 50 (reduced from paper's 1000 for runtime)
- Track contrastive loss, temperature
- No labels used during pretraining

### 9. Linear Evaluation
- Freeze encoder weights
- Train a single linear layer (512 → 10) on labeled CIFAR-10
- 20 epochs, Adam optimizer
- Report accuracy on test set

### 10. From-Scratch Baseline
- Train the same ResNet-18 from scratch on labeled CIFAR-10 (supervised)
- Same epochs, same optimizer
- Compare test accuracy with SimCLR linear eval

### 11. Visualization & Comparison
- Plot: contrastive loss curve during pretraining
- Plot: linear eval accuracy vs from-scratch accuracy
- t-SNE of learned representations (optional, if time permits)

## Key Functions/Classes

| Component | Class/Function | Input → Output |
|---|---|---|
| Augmentation | `SimCLRAugment` | PIL Image → Tensor |
| Dataset | `ContrastiveDataset` | index → (view1, view2, label) |
| Encoder | `ResNet18` (torchvision) | Image tensor (B,3,32,32) → features (B,512) |
| Projection | `ProjectionHead` | (B,512) → (B,128) |
| Loss | `nt_xent_loss` | (B,128), (B,128) → scalar |
| Model | `SimCLR` | (view1, view2) → (z1, z2) |
| Linear Eval | `LinearClassifier` | (B,512) → (B,10) |

## Data Flow & Shapes

```
Image (3, 32, 32)
  → augment → view_i (3, 32, 32), view_j (3, 32, 32)
  → encoder f(·) → h_i (512), h_j (512)
  → projection g(·) → z_i (128), z_j (128)
  → NT-Xent loss → scalar
```

Batch of N images produces 2N views, similarity matrix is 2N × 2N.

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|---|---|---|
| Dataset | ImageNet (1.28M images) | CIFAR-10 (50K images) |
| Encoder | ResNet-50 (24M params) | ResNet-18 (~1M params) |
| Epochs | 1000 | 50 |
| Batch size | 4096 | 256 |
| Projection dim | 128 | 128 |
| Temperature | 0.5 (learned in v2) | 0.5 (fixed) |
| Augmentations | 10+ compositions | Core 5 (crop, jitter, grayscale, blur, flip) |
| No memory bank | Correct (SimCLR doesn't use one) | Same |
| Global batch norm | Yes | Local batch norm (single GPU) |
