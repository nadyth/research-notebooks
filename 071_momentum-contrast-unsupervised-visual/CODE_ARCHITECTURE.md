# Code Architecture — MoCo Notebook

## Overview

The notebook implements MoCo (Momentum Contrast) for unsupervised visual representation learning on a small-scale dataset (CIFAR-10 or STL-10), following the paper's core design: a query encoder, a momentum-updated key encoder, a queue of negative keys, and InfoNCE contrastive loss. It then evaluates the learned representations via linear probing and compares against a SimCLR-style in-batch-negative baseline.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, torchvision, numpy, matplotlib
- Device selection (GPU if available, else CPU)
- Random seed for reproducibility

### 2. Data Preparation & Augmentation
- **MoCoAugment transform:** two random crops with color jitter (brightness, contrast, saturation, hue), random horizontal flip, random grayscale — produces two augmented views (x_q, x_k) of each image
- Dataset: CIFAR-10 (unlabeled split for training; small labeled split for linear eval)
- Two dataloaders: `unlabeled_loader` (with MoCo augmentation) and `labeled_loader` (standard normalization for linear eval)

### 3. Model Architecture
- **Encoder backbone:** ResNet-18 (modified for CIFAR: first conv 3×3, no maxpool) — shared architecture for both f_q and f_k
- **Projection head:** 2-layer MLP (256→128) with ReLU, appended to the backbone features
- **MoCo wrapper class `MoCo`:**
  - `__init__`: creates f_q (encoder_q + head_q) and f_k (encoder_k + head_k), initializes f_k = f_q (same weights), creates queue tensor of shape [128, K] filled with random unit vectors, sets m=0.999
  - `forward(x_q, x_k)`:
    1. Encode queries: q = f_q(x_q) → normalize
    2. Compute keys with no_grad: k = f_k(x_k) → normalize
    3. Compute logits: l_positive = q·k+ (diagonal), l_negative = q·queue → form logits = cat([positive, negatives]) / temperature
    4. InfoNCE loss: CrossEntropy with target = 0 (positive is first)
    5. Dequeue & enqueue: roll queue, insert current batch keys
  - `momentum_update`: θ_k ← m·θ_k + (1-m)·θ_q (called after optimizer.step())

### 4. SimCLR Baseline (in-batch negatives)
- **SimCLR wrapper class:** same encoder + projection head, but uses all other samples in the batch as negatives
- Computes loss within the batch (no queue, no momentum encoder)
- Provided for comparison with MoCo's queue-based approach

### 5. Training Loop
- For each epoch:
  - Load batch, create two augmented views
  - Forward through MoCo → loss
  - Backprop → optimizer step (SGD with cosine LR schedule, lr=0.03×batch/256, momentum=0.9, weight_decay=1e-4)
  - Momentum update key encoder
- Log loss every N steps
- Training runs for ~50 epochs (reduced from paper's 200 for Kaggle runtime)

### 6. Linear Evaluation
- Freeze the trained f_q encoder
- Train a linear classifier on top of frozen features using the labeled CIFAR-10 subset
- Report accuracy on test set
- Also evaluate SimCLR baseline the same way for comparison

### 7. Visualization & Comparison
- Plot training loss curves (MoCo vs SimCLR)
- Plot linear eval accuracy comparison
- t-SNE of learned features (optional, if time permits)
- Print summary table: Method | Queue Size | Linear Acc

## Key Functions/Classes

| Component | Purpose |
|-----------|---------|
| `MoCoAugment` | Produces two augmented views for contrastive pairs |
| `ResNet18CIFAR` | ResNet-18 adapted for 32×32 CIFAR images |
| `MoCo` | Main model: query/key encoders, queue, momentum update, InfoNCE loss |
| `SimCLR` | Baseline model using in-batch negatives only |
| `train_moco()` | Training loop with momentum update and queue management |
| `train_simclr()` | Training loop for SimCLR baseline |
| `linear_eval()` | Trains linear classifier on frozen features, returns accuracy |

## Data Flow / Tensor Shapes

```
Input batch: [B, 3, 32, 32]
  → augment → x_q: [B, 3, 32, 32], x_k: [B, 3, 32, 32]
  → f_q(x_q) → q: [B, 128] (normalized)
  → f_k(x_k) → k: [B, 128] (normalized, no_grad)
  → logits = cat([q·k+ diag, q·queue]) / 0.07 → [B, 1+K]
  → loss = CrossEntropy(logits, zeros)
  → queue update: queue = cat([k, queue[:-B]], dim=1) → [128, K]
```

## Deliberate Simplifications vs Full Paper

| Paper | Notebook |
|-------|----------|
| ImageNet, ResNet-50 | CIFAR-10, ResNet-18 |
| Queue size K=65536 | K=4096 (memory-friendly) |
| 200 epochs | 50 epochs (Kaggle runtime) |
| MLP projection dim=128 | Same (128) |
| MoCo v2 additions ( projection head on both, mlp projection, stronger aug) | MoCo v1 style (linear projection, standard aug) — MoCo v2 enhancements kept minimal |
| Multi-GPU training | Single GPU |
| Full ImageNet linear eval | CIFAR-10 linear eval with small labeled subset |
| Shuffling BN trick (distributed) | Simplified — single GPU, standard BN |
