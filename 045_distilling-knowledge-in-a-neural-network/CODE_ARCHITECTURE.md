# Code Architecture — Knowledge Distillation Notebook

## Overview

The notebook implements knowledge distillation from Hinton et al. (2015) from scratch in PyTorch. A large teacher MLP is trained on MNIST, then a smaller student MLP is trained three ways: (1) from hard labels only, (2) from the teacher's soft targets via distillation, and (3) from a weighted combination of soft and hard targets. The three students are compared on test accuracy, and the teacher's soft-target distributions are visualized to show the "dark knowledge" being transferred. All computation is CPU-friendly (small MLPs on MNIST, no GPU required).

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `torch.nn.functional`, `torchvision` for MNIST
- `numpy`, `matplotlib` for visualization
- Random seed fixing for reproducibility
- Device set to CPU (this is a CPU-friendly notebook)

### 2. Data Loading — MNIST
- Load MNIST train/test sets via `torchvision.datasets.MNIST`
- Normalize to [0, 1] with mean/std normalization
- DataLoaders with batch_size=128 for training, 256 for testing
- Data shapes: train images (60000, 1, 28, 28), train labels (60000,)

### 3. Teacher Network (Large MLP)
- **Class `TeacherNet(nn.Module)`**
  - Input: 784 (flattened 28×28)
  - Hidden: two layers of 800 ReLU units (matching the paper's "small net" baseline)
  - Dropout (0.5) on hidden layers — the paper used dropout + weight constraints as regularization
  - Output: 10 logits (MNIST digits)
  - Forward: 784 → 800 → Dropout → 800 → Dropout → 10
  - ~1.3M parameters

### 4. Student Network (Small MLP)
- **Class `StudentNet(nn.Module)`**
  - Input: 784
  - Hidden: two layers of 300 ReLU units (smaller than teacher — the paper used 300+ units with good results)
  - No dropout (we want to show distillation regularizes the student)
  - Output: 10 logits
  - Forward: 784 → 300 → 300 → 10
  - ~267K parameters (5x smaller than teacher)

### 5. Softmax with Temperature — From Scratch
- **Function `softmax_with_temperature(logits, T)`**
  - Input: logits (N, 10), temperature T (scalar)
  - Compute: exp(logits / T) / sum(exp(logits / T), dim=1, keepdim=True)
  - Returns soft probability distribution (N, 10)
  - Demonstrates: at T=1, standard softmax; at T=10, much softer distribution

### 6. Distillation Loss — From Scratch
- **Function `distillation_loss(student_logits, teacher_logits, T)`**
  - Compute soft targets: teacher_soft = softmax_with_temperature(teacher_logits, T)
  - Compute student soft: student_soft = softmax_with_temperature(student_logits, T)
  - Loss: KL divergence = sum(teacher_soft * log(teacher_soft / student_soft))
  - Equivalent to cross-entropy between teacher soft targets and student soft predictions
  - Multiply by T^2 to compensate for gradient scaling (as specified in the paper)
  - Key shapes: student_logits (B, 10), teacher_logits (B, 10) → loss (scalar)

### 7. Combined Loss Function
- **Function `combined_loss(student_logits, teacher_logits, labels, T, alpha)`**
  - Soft loss: alpha * T^2 * KL(teacher_soft || student_soft)
  - Hard loss: (1 - alpha) * CrossEntropy(student_logits, labels)
  - alpha controls the trade-off (paper recommends alpha ~0.7-0.9)
  - Returns total loss

### 8. Training the Teacher
- Train TeacherNet on MNIST for 10 epochs
- Optimizer: Adam, lr=0.001
- Loss: standard CrossEntropy with hard labels
- Record training loss, test accuracy per epoch
- Expected: ~98% test accuracy

### 9. Training Students — Three Conditions
- **Student-Hard**: trained with standard CrossEntropy on hard labels only
- **Student-Distill**: trained with distillation loss only (soft targets, T=4)
- **Student-Combined**: trained with combined loss (alpha=0.7, T=4)
- All three use identical architecture (StudentNet), same data, same optimizer (Adam, lr=0.001)
- Train each for 10 epochs
- Record training loss, test accuracy per epoch for comparison

### 10. Q-Value Overestimation Analysis (Dark Knowledge Visualization)
- For a sample of test images, extract the teacher's soft predictions at T=1 and T=10
- Show that the teacher assigns small but non-trivial probabilities to "wrong" classes
- Example: a handwritten "2" might get 99.7% for 2, 0.2% for 3, 0.05% for 7
- These ratios carry the similarity structure ("which 2s look like 3s")
- Visualize as heatmaps: teacher soft probs for several test images

### 11. Temperature Sensitivity Experiment
- Train separate distilled students at T = [1, 2, 4, 8, 16, 20]
- Plot final test accuracy vs. temperature
- Show that intermediate temperatures (2-8) work best for this student size
- (Paper: large students tolerate high T; very small students prefer lower T)

### 12. Comparison of Soft vs Hard Target Gradients
- For a single training example, compute the gradient magnitude from:
  - Hard label loss (one-hot target)
  - Soft target loss (teacher's softened distribution at T=4)
- Show that soft targets produce more informative, lower-variance gradients
- This explains why distilled students can train faster with fewer examples

### 13. Visualization & Results
- Plot 1: Training loss curves (teacher, student-hard, student-distill, student-combined)
- Plot 2: Test accuracy curves over epochs for all three student conditions
- Plot 3: Teacher soft target distribution for sample images (bar charts at T=1 vs T=10)
- Plot 4: Temperature sensitivity — accuracy vs. temperature
- Plot 5: Confusion matrices for student-hard vs student-distill

### 14. Summary & Analysis
- Print final test accuracies for all models
- Show that distillation significantly improves student accuracy
- Discuss: dark knowledge transfer, temperature effects, gradient analysis
- Compare parameter counts: teacher vs student

## Key Functions/Classes
| Component | Purpose |
|---|---|
| `TeacherNet` | Large MLP (800-800-10) with dropout |
| `StudentNet` | Small MLP (300-300-10) without dropout |
| `softmax_with_temperature()` | Temperature-scaled softmax from scratch |
| `distillation_loss()` | KL-divergence-based soft target loss |
| `combined_loss()` | Weighted combination of soft + hard losses |
| `train_model()` | Generic training loop supporting any loss function |
| `evaluate_model()` | Test accuracy evaluation |

## Data Flow & Shapes
```
MNIST image: (1, 28, 28) → flatten → (784,)
  ↓ TeacherNet
  teacher_logits: (10,) → softmax(T=4) → teacher_soft: (10,)
  ↓ StudentNet
  student_logits: (10,) → softmax(T=4) → student_soft: (10,)
  ↓
  distillation_loss = T^2 * KL(teacher_soft || student_soft)
  hard_loss = CrossEntropy(student_logits, true_label)
  total = alpha * distillation_loss + (1-alpha) * hard_loss
```

## Deliberate Simplifications vs Full Paper
1. **MNIST only** — paper also demonstrated on large-scale speech recognition (85M param acoustic model) and JFT image dataset; we use MNIST for CPU tractability
2. **Single teacher** — paper distilled from ensembles of 10 models; we use a single large model (the paper notes single-model distillation also works well)
3. **Small MLPs** — paper used 1200-unit teacher and 800-unit student; we use 800/300 for faster CPU training
4. **No specialist models** — paper introduced ensembles of specialists for JFT; we focus on the core distillation mechanism
5. **No transfer-set experiments** — paper showed distillation works even when transfer set omits entire classes; we use the full training set
6. **Fixed alpha** — paper explored various alpha values; we use alpha=0.7 as a representative value
7. **No speech/ASR experiments** — the Android voice search experiments require proprietary data and massive compute
