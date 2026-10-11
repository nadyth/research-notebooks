# Code Architecture — DistilBERT Knowledge Distillation Notebook

## Overview

The notebook demonstrates knowledge distillation from a teacher Transformer model (BERT-style) to a shallower student model, following the DistilBERT methodology. It compresses a small pretrained Transformer into a 2-layer student (vs the teacher's 4-layer) and compares accuracy and inference speed on a text classification task (IMDB sentiment).

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `transformers`, `datasets`, `torch`, `scikit-learn`, `matplotlib`.
- Imports PyTorch, HuggingFace transformers, datasets, and utility libraries.

### 2. Data Loading & Preprocessing
- Loads the IMDB movie review dataset (binary sentiment classification) via `datasets.load_dataset("imdb")`.
- Subsamples to 2000 train / 500 test examples for speed (the full 50K dataset would take too long).
- Tokenizes with the teacher model's tokenizer (`bert-base-uncased`).
- Creates PyTorch DataLoaders with batch size 16.

### 3. Teacher Model
- Loads `bert-base-uncased` as the teacher (frozen weights, eval mode only).
- The teacher is NOT trained — it's used purely to generate soft target logits for distillation.
- Teacher has 4 Transformer layers (config: `num_hidden_layers=4`, `hidden_size=256`, 4 attention heads) — a toy-scale BERT to keep runtime under 30 min on GPU.

### 4. Student Model
- Creates a custom student config with 2 Transformer layers (half the teacher), same hidden size.
- The student is initialized from the teacher's layers (copies layer indices 0 and 2 from the teacher).
- Student has ~50% of the teacher's parameters.
- A classification head on top of the [CLS] token's final hidden state.

### 5. Distillation Training Loop
- **Triple loss** (following the paper):
  1. `distillation_loss = T² · KL(softmax(teacher_logits/T) || softmax(student_logits/T))` — KL divergence between temperature-scaled soft targets. Temperature T=2.
  2. `ce_loss = CrossEntropy(student_logits, hard_labels)` — standard classification loss on ground-truth labels.
  3. `cosine_loss = 1 - cos(student_hidden, teacher_hidden)` — cosine embedding loss on the [CLS] hidden states.
- Total loss = α · distillation_loss + β · ce_loss + γ · cosine_loss (α=0.5, β=0.5, γ=0.1).
- Optimizer: AdamW with learning rate 2e-5, weight decay 0.01.
- Trained for 3 epochs on the subsampled IMDB data.
- Teacher is in eval mode (no gradient); student is trained.

### 6. Evaluation & Comparison
- Evaluates student vs teacher on the test set: accuracy, F1 score.
- Measures inference speed: time per batch for student vs teacher (forward pass only, batch size 16).
- Prints parameter counts: teacher vs student.
- Plots: (a) parameter count comparison bar chart, (b) inference speed comparison, (c) training loss curve.

### 7. Results Summary
- Prints a formatted table: Model | Params | Accuracy | F1 | Inference (ms/batch)
- Shows that the student retains most of the teacher's accuracy with ~50% fewer parameters and ~2x faster inference.

## Key Functions/Classes

| Function/Class | Purpose |
|---|---|
| `get_teacher_logits(batch)` | Runs the frozen teacher model to get soft target logits for distillation |
| `distillation_loss_fn(student_logits, teacher_logits, T)` | Computes KL divergence with temperature scaling |
| `StudentModel` | A `BertForSequenceClassification` with fewer layers, initialized from teacher |
| `train_distillation(teacher, student, train_loader, epochs)` | Main training loop implementing the triple loss |
| `evaluate_model(model, test_loader)` | Computes accuracy and F1 on the test set |
| `measure_inference_speed(model, test_loader)` | Times forward passes to compare student vs teacher speed |

## Data Flow / Shapes

```
Input text → Tokenizer (BERT tokenizer) → input_ids [B, 512]
                                         → attention_mask [B, 512]
    → Teacher (frozen, 4 layers) → teacher_logits [B, 2], teacher_hidden [B, 256]
    → Student (trainable, 2 layers) → student_logits [B, 2], student_hidden [B, 256]
    → Triple loss(teacher_logits, student_logits, teacher_hidden, student_hidden, labels)
    → Backprop through student only
```

## Deliberate Simplifications vs Full Paper

1. **Toy-scale models**: Instead of BERT-base (12 layers, 110M params), we use a 4-layer teacher and 2-layer student with hidden size 256. This keeps Kaggle GPU runtime under 30 minutes.
2. **Task-level distillation**: The paper distills during pre-training (MLM on Wikipedia). We distill during fine-tuning on a downstream task (IMDB classification), which is more practical and faster to demonstrate. The triple loss structure (distillation + CE + cosine) faithfully follows the paper.
3. **Subsampled data**: 2000 train / 500 test examples vs the full 50K IMDB dataset, for speed.
4. **No MLM loss during distillation**: The paper includes an MLM loss during pre-training distillation. Since we do task-level distillation, we use cross-entropy on labels instead, which is the standard adaptation.
5. **Temperature**: We use T=2 as in the paper. The KL loss is scaled by T² as per Hinton et al. (2015).
6. **No linear layer initialization trick**: The paper initializes the student from every other teacher layer. We do the same for the Transformer layers but use a fresh classification head.
