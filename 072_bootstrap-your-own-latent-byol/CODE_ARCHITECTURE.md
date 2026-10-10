# Code Architecture — BYOL Notebook

## Overview

The notebook implements BYOL (Bootstrap Your Own Latent) for self-supervised image representation learning on a small-scale synthetic dataset, following the paper's core design: an online network (encoder + projector + predictor) and a target network (encoder + projector, EMA-updated), trained with a simple MSE loss between normalized predictions and targets — no negative pairs. It then evaluates the learned representations via linear probing and verifies that representations do not collapse via embedding-variance plots.

## Section-by-Section Breakdown

### 1. Setup & Configuration
- PyTorch, numpy, matplotlib
- Device selection (GPU if available)
- Hyperparameters: batch size 128, epochs 8, feature dim 256, projection dim 128, EMA decay 0.99→1.0 cosine schedule, LR 0.03
- Random seed for reproducibility

### 2. Data Preparation & Augmentation
- **BYOLAugment transform:** produces two augmented views (v, v′) of each image using random crop, color jitter, horizontal flip, random grayscale, and Gaussian blur — the first view gets stronger augmentation (asymmetric, per paper)
- **SyntheticImageDataset:** structured synthetic images with per-class color patterns (same as MoCo notebook pattern) so the encoder has learnable structure
- Two dataloaders: unlabeled (augmented pairs) for pre-training, labeled (standard) for linear eval

### 3. Model Architecture
- **SmallCNNEncoder:** 3 conv blocks + global avg pool + linear → 256-dim feature (replaces ResNet-50 for tractable runtime)
- **ProjectionHead:** 2-layer MLP (256→256→128) with BatchNorm + ReLU (BatchNorm is critical per paper ablation)
- **PredictorHead:** 2-layer MLP (128→256→128) with BatchNorm + ReLU — the key component that prevents collapse
- **BYOL class:**
  - `__init__`: creates online (encoder + projector + predictor) and target (encoder + projector), initializes target = online (same weights), sets `target_ema = 0.996`
  - `forward(v, v′)`:
    1. Online: z = projector(encoder(v)), p = predictor(z) → normalize p
    2. Target (no_grad): z′ = projector_target(encoder_target(v′)) → normalize z′
    3. Loss = MSE(p_normalized, z′_normalized) = 2 − 2·⟨p, z′⟩
  - `update_target()`: ξ ← τ·ξ + (1−τ)·θ for all target params, with τ on cosine schedule

### 4. SimSiam-style Baseline (Stop-gradient + Predictor, No EMA)
- Same encoder + projector + predictor architecture, but target network = online network (shared weights, stop-gradient only, no EMA)
- Demonstrates that the predictor + stop-gradient alone (without EMA) also works — the SimSiam insight

### 5. Training Loop (BYOL)
- For each epoch:
  - Update EMA decay: τ = 1 − (1 − 0.996)·(cos(π·epoch/total) + 1)/2 (cosine schedule from 0.996→1.0)
  - Load batch, create two augmented views
  - Forward through BYOL → MSE loss
  - Backprop through online network only → SGD with cosine LR
  - EMA update of target network
- Log loss every epoch
- Track embedding variance per epoch (to verify no collapse)

### 6. Training SimSiam Baseline
- Same training loop but with stop-gradient only (no EMA target)
- For comparison with BYOL

### 7. Linear Evaluation
- Freeze the trained online encoder
- Train a linear classifier on top of frozen features using labeled data
- Report accuracy on test set for both BYOL and SimSiam

### 8. Visualization & Collapse Check
- **Training loss curves:** BYOL vs SimSiam
- **Linear eval accuracy:** bar chart comparison
- **Embedding variance plot:** tracks the variance of the encoder's output embeddings across training epochs — if embeddings collapse, variance → 0. A healthy BYOL run maintains high variance.
- **t-SNE / PCA scatter:** 2D projection of learned features colored by class label — should show class-separated clusters if representations are meaningful

### 9. Summary & Discussion
- Print summary table: Method | Linear Acc | Embedding Variance | Collapsed?
- Discuss what would happen without the predictor (collapse)
- Note the difference between BYOL (EMA target) and SimSiam (stop-gradient only)

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `SmallCNNEncoder` | Compact CNN backbone (replaces ResNet-50) |
| `ProjectionHead` | MLP with BatchNorm — maps features to projection space |
| `PredictorHead` | MLP with BatchNorm — maps online projection to target space; KEY anti-collapse component |
| `BYOL` | Main model: online + target networks, forward returns BYOL loss, `update_target()` does EMA |
| `SimSiamBaseline` | Stop-gradient + predictor without EMA, for comparison |
| `BYOLAugment` | Produces two asymmetric augmented views of each image |
| `train_byol()` | Training loop with EMA target update + cosine LR + variance tracking |
| `linear_eval()` | Frozen encoder + linear classifier training and evaluation |
| `check_collapse()` | Computes embedding variance to verify non-collapse |

## Data Flow / Shapes

```
Image x [3, 32, 32]
  → Augment → v [3, 32, 32], v′ [3, 32, 32]
  → Encoder(v) → y_θ [256]        Encoder_target(v′) → y_ξ [256]
  → Projector(y_θ) → z_θ [128]    Projector_target(y_ξ) → z_ξ [128]
  → Predictor(z_θ) → p [128]
  → normalize(p), normalize(z_ξ)
  → Loss = MSE(normalize(p), normalize(z_ξ))  [scalar]
```

## Deliberate Simplifications vs Full Paper

1. **Backbone**: SmallCNN (~0.4M params) instead of ResNet-50 (~25M). Paper uses ResNet-50/ResNet-large.
2. **Dataset**: Synthetic structured images instead of ImageNet (1.28M images). Paper trains on full ImageNet.
3. **Epochs**: 8 epochs instead of 1000. Paper trains for 1000 epochs with 4096 batch size.
4. **Optimizer**: SGD instead of LARS. Paper uses LARS with specific weight decay schedule.
5. **Augmentation**: Simplified augmentation pipeline (crop + jitter + flip + grayscale + blur) instead of the full BYOL recipe (which also includes solarization, multi-crop in some variants).
6. **EMA schedule**: Cosine from 0.996 to 1.0 (matching paper), but compressed into fewer epochs.
7. **Batch size**: 128 instead of 4096. Paper requires large batches for best performance.
8. **No transfer evaluation**: Paper evaluates on transfer benchmarks (VOC, COCO, etc.); we only do linear probing.
