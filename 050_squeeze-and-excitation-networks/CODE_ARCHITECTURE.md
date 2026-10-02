# Code Architecture — Squeeze-and-Excitation Networks Notebook

## Notebook Structure

### 1. Setup & Imports
- PyTorch, torchvision, matplotlib, numpy
- Device selection (GPU if available, else CPU)
- Reproducibility: fixed random seeds

### 2. SE Block Implementation (`SEBlock` class)
- **Input:** Feature map tensor `U ∈ ℝ^(B, C, H, W)`
- **Squeeze:** `nn.AdaptiveAvgPool2d(1)` → `z ∈ ℝ^(B, C, 1, 1)` then squeeze to `(B, C)`
- **Excitation:** FC bottleneck: `Linear(C, C//r)` → ReLU → `Linear(C//r, C)` → Sigmoid
  - Implemented as `nn.Conv2d(C, C//r, 1)` and `nn.Conv2d(C//r, C, 1)` (1×1 convs = FC layers for spatial data)
  - Reduction ratio `r=16`
- **Scale:** `s.unsqueeze(-1).unsqueeze(-1) * U` → channel-wise multiplication
- **Output:** `x̃ ∈ ℝ^(B, C, H, W)` — recalibrated feature map
- Data flow: `(B, C, H, W) → pool → (B, C, 1, 1) → squeeze → (B, C) → FC1 → (B, C//r) → ReLU → FC2 → (B, C) → sigmoid → (B, C, 1, 1) → multiply → (B, C, H, W)`

### 3. Baseline CNN (`BaselineCNN` class)
- Small conv stack: 3 conv blocks (conv → BN → ReLU → maxpool), then global avg pool → FC classifier
- Input: `(B, 3, 32, 32)` (CIFAR-10 compatible)
- Channels: 32 → 64 → 128
- Output: logits for 10 classes

### 4. SE-Enhanced CNN (`SECNN` class)
- Identical architecture to BaselineCNN but with `SEBlock` inserted after each conv block's BN+ReLU
- Same channel progression: 32 → 64 → 128
- SE blocks with r=16 at each stage
- Allows direct comparison to baseline

### 5. Data Loading
- CIFAR-10 via torchvision
- Standard augmentations: random crop, random horizontal flip, normalize
- Batch size: 128

### 6. Training Loop
- Optimizer: SGD with momentum=0.9, weight_decay=5e-4
- Scheduler: CosineAnnealingLR
- Loss: CrossEntropyLoss
- Epochs: 15 (kept low for Kaggle runtime)
- Train both baseline and SE models in the same loop, track losses/accuracies

### 7. Channel Attention Visualization
- Load a few test images
- Pass through SE model, hook the SE block excitation weights (sigmoid outputs)
- Plot: input image + bar chart of channel weights for the first SE block
- Shows which channels get up-weighted (s→1) vs down-weighted (s→0) per image
- Demonstrates that different images produce different excitation patterns

### 8. Comparison & Results
- Plot training curves (loss & accuracy) for baseline vs SE model
- Print final test accuracy for both models
- Show parameter count comparison (SE adds ~small % overhead)
- Bar chart: baseline accuracy vs SE accuracy

## Key Shapes

| Stage | Shape (B=128) |
|-------|--------------|
| Input | (128, 3, 32, 32) |
| After Conv1+BN+ReLU+SE | (128, 32, 32, 32) |
| After Pool1 | (128, 32, 16, 16) |
| After Conv2+BN+ReLU+SE | (128, 64, 16, 16) |
| After Pool2 | (128, 64, 8, 8) |
| After Conv3+BN+ReLU+SE | (128, 128, 8, 8) |
| After Pool3 | (128, 128, 4, 4) |
| After Global Avg Pool | (128, 128) |
| After FC | (128, 10) |

## Deliberate Simplifications vs Full Paper

1. **Scale:** Full paper uses ResNet-50/101 on ImageNet (224×224, 1000 classes). We use a small 3-block CNN on CIFAR-10 (32×32, 10 classes) for tractable runtime.
2. **Reduction ratio:** Paper explores r ∈ {2,4,8,16,32}; we fix r=16 (the paper's recommended default).
3. **Architecture:** Paper integrates SE into ResNet, ResNeXt, Inception, VGG. We use a simple custom CNN to isolate the SE block's effect.
4. **Ablations:** Paper performs extensive ablations (excitation operator, integration stage, etc.). We include only the channel visualization as qualitative analysis.
5. **Training:** Paper trains for ~100 epochs with heavy augmentation. We train 15 epochs with basic augmentation for Kaggle runtime constraints.
6. **Placeholders:** No pretrained backbone — everything trained from scratch on CIFAR-10.
