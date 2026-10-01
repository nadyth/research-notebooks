# Code Architecture — Label Smoothing Notebook

## Overview

The notebook implements Label Smoothing Regularization (LSR) from scratch in PyTorch, trains a small image classifier with and without it on Fashion-MNIST, and compares calibration (confidence histograms, reliability diagrams) between the two models. The implementation is GPU-friendly but also runs on CPU within a reasonable time.

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `torch.nn.functional`, `torchvision`, `matplotlib`, `numpy`
- Device detection (CUDA if available, else CPU)
- Random seed fixing for reproducibility

### 2. Label Smoothing Loss from Scratch
- **Class `LabelSmoothingLoss(nn.Module)`**
  - `__init__(num_classes, smoothing=0.1)`: stores ε and K
  - `forward(logits, target)`:
    - Compute standard log-softmax: `log_probs = F.log_softmax(logits, dim=-1)`
    - Create smoothed target distribution: `q' = (1-ε) * one_hot + ε/K`
    - Compute cross-entropy against the smoothed target: `loss = -sum(q' * log_probs) / batch_size`
    - Key shapes: logits (B, K), target (B,), log_probs (B, K), one_hot (B, K), loss scalar
  - Alternative implementation via `F.kl_div` is also shown for comparison

### 3. NumPy Reference Implementation
- **Function `label_smoothing_numpy(logits, target, num_classes, epsilon=0.1)`**
  - Demonstrates the same computation in pure NumPy for pedagogical clarity
  - Shows the smoothed target matrix explicitly
  - Verifies the equivalence: loss = (1-ε)·CE_hard + ε·CE_uniform

### 4. Dataset Preparation
- **Fashion-MNIST** via torchvision (28×28 grayscale, 10 classes)
  - 60,000 train images, 10,000 test images
  - Normalized with mean=0.2860, std=0.3530 (Fashion-MNIST statistics)
  - DataLoaders with batch_size=128

### 5. Model Architecture
- **Class `SmallCNN(nn.Module)`**
  - Two conv blocks: [Conv2d(1→32, 3×3) → ReLU → MaxPool2d(2×2)] × 2
  - Feature dimension: 28→28→14→14→7→7 → flatten to 32×7×7 = 1568
  - FC layer: 1568 → 128 → ReLU → Dropout(0.3) → 10
  - Output: raw logits (B, 10)
  - Total parameters: ~220K — small enough to train quickly on GPU or CPU

### 6. Training Function
- **Function `train_model(model, train_loader, criterion, optimizer, epochs, device)`**
  - Standard training loop with per-epoch loss/accuracy tracking
  - Returns training history (losses, accuracies)
  - Supports both standard CrossEntropyLoss and LabelSmoothingLoss as criterion

### 7. Evaluation & Calibration Metrics
- **Function `evaluate_with_confidence(model, test_loader, device)`**
  - Collects: predicted probabilities (softmax), predicted class, true class for all test samples
  - Computes: test accuracy, mean confidence, expected calibration error (ECE)
  - ECE: bins predictions by confidence (10 bins), computes |accuracy(bin) - confidence(bin)| weighted by bin size
  - Returns confidence values, correctness array, ECE

### 8. Training Two Models
- Model A: trained with standard `nn.CrossEntropyLoss()` (hard one-hot labels)
- Model B: trained with `LabelSmoothingLoss(num_classes=10, smoothing=0.1)`
- Same architecture, same optimizer (Adam, lr=1e-3), same number of epochs (10)
- Both models trained on the same data split for fair comparison

### 9. Visualization & Comparison
- **Confidence histograms**: distribution of max-softmax probabilities for correct vs incorrect predictions, side by side for both models
- **Reliability diagram**: 10-bin calibration plot — ideal is a diagonal line; shows how far each model deviates
- **Accuracy vs confidence bar chart**: per-bin accuracy and confidence for both models
- **Training curves**: loss and accuracy over epochs for both models

### 10. Summary Statistics
- Print table: accuracy, mean confidence, ECE for both models
- Discussion: label smoothing should produce slightly lower confidence but better calibration (lower ECE)

## Data Flow / Shapes

```
Input: (B, 1, 28, 28) images
  → Conv1: (B, 32, 28, 28) → ReLU → MaxPool: (B, 32, 14, 14)
  → Conv2: (B, 32, 14, 14) → ReLU → MaxPool: (B, 32, 7, 7)
  → Flatten: (B, 1568)
  → FC1: (B, 128) → ReLU → Dropout
  → FC2: (B, 10) [logits]
  → Softmax: (B, 10) [probabilities]

LabelSmoothingLoss:
  logits (B, 10) + target (B,) → log_softmax (B, 10)
  → one_hot (B, 10) → smooth: q' = (1-ε)*one_hot + ε/K (B, 10)
  → loss = -mean(sum(q' * log_softmax, dim=1))
```

## Deliberate Simplifications vs. Full Paper

1. **Fashion-MNIST instead of ImageNet (ILSVRC 2012)**: The paper uses 1000-class ImageNet with ~1.2M images. We use 10-class Fashion-MNIST with 60K images for faster training. The label smoothing formula is identical.
2. **Small CNN instead of Inception-v3**: The paper's main contribution is the Inception-v3 architecture; we focus exclusively on the label smoothing regularizer and use a simple CNN as the classifier. This isolates the effect of label smoothing.
3. **ε = 0.1 (same as paper)**: We use the same smoothing value as the paper's ImageNet experiments.
4. **10 epochs instead of 100**: The paper trains for 100 epochs on 50 GPUs. We train for 10 epochs on a single GPU/CPU — enough to show the calibration difference.
5. **ECE instead of top-1/top-5 error**: The paper reports top-1/top-5 classification error. We additionally compute Expected Calibration Error (ECE) to directly measure the calibration improvement that label smoothing provides.
6. **No ensemble or multi-crop evaluation**: The paper uses ensembles of 4 models and multi-crop evaluation. We use single-model, single-crop evaluation for simplicity.
