# Code Architecture: LoRA Notebook

## Overview

The notebook implements LoRA (Low-Rank Adaptation) from scratch in PyTorch, demonstrating the technique on a small Transformer model for a text classification task. It compares full fine-tuning vs. LoRA fine-tuning in terms of trainable parameters, training speed, and final accuracy.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import PyTorch, torch.nn, and standard utilities.
- Set random seeds for reproducibility.
- Configure device (CPU/GPU).

### 2. Toy Dataset Preparation
- Create a small synthetic text classification dataset (e.g., sentiment-like binary classification with short sequences).
- Build a simple vocabulary and tokenize inputs into integer ID sequences.
- Create train/validation DataLoader objects.
- **Data shapes**: Input tensors are `(batch_size, seq_len)` with integer token IDs; labels are `(batch_size,)`.

### 3. Small Transformer Model (Pretrained Base)
- Implement a minimal Transformer encoder from scratch: token embeddings + positional encoding + N Transformer encoder layers (multi-head self-attention + feedforward + residual connections + layer norm).
- **Key class**: `SimpleTransformer(nn.Module)` with configurable `d_model`, `n_heads`, `n_layers`, `vocab_size`, `num_classes`.
- This simulates the "pre-trained" model — we initialize it and do a light pre-training pass so it has non-random weights, then freeze it.
- **Forward pass shapes**: `(batch, seq_len)` → embedding `(batch, seq_len, d_model)` → encoder layers `(batch, seq_len, d_model)` → pooled → classifier `(batch, num_classes)`.

### 4. LoRA Layer Implementation
- **Key class**: `LoRALinear(nn.Module)` — wraps a frozen `nn.Linear` layer and adds trainable low-rank matrices A and B.
  - Original weight: `W₀` (frozen, `requires_grad=False`)
  - LoRA matrices: `A` (r×k), `B` (d×r) — trainable
  - Initialization: A ~ N(0, σ²), B = 0
  - Forward: `h = W₀·x + (α/r)·B·A·x`
  - `merge()` method: computes `W = W₀ + (α/r)·B·A` and replaces the frozen weight (for zero-latency inference).
- **Shapes**: If original Linear is `(d_model, d_model)`, then A is `(r, d_model)`, B is `(d_model, r)`, with r = 4 or 8.

### 5. Injecting LoRA into the Transformer
- Function `apply_lora_to_model(model, r, alpha, target_modules)` that replaces specified Linear layers (Wq, Wv) with LoRALinear wrappers.
- The original model parameters are frozen (`requires_grad=False`), only LoRA A/B matrices are trainable.
- Function `count_trainable_parameters(model)` to report the difference.

### 6. Full Fine-Tuning Baseline
- Clone the pre-trained model, keep all parameters trainable.
- Train with Adam optimizer on the classification task for N epochs.
- Record training loss, validation accuracy, and wall-clock time.

### 7. LoRA Fine-Tuning
- Apply LoRA to the pre-trained model (r=4, α=4, on Wq and Wv).
- Train only the LoRA parameters with Adam on the same task for the same N epochs.
- Record training loss, validation accuracy, and wall-clock time.

### 8. Comparison & Analysis
- Compare: total parameters, trainable parameters (full FT vs LoRA), training time, final accuracy.
- Print a summary table.
- Demonstrate LoRA merge: call `merge()` on the LoRA model and verify forward pass produces identical outputs.

### 9. Rank Ablation (Optional)
- Try r = 1, 2, 4, 8, 16 and plot validation accuracy vs. rank.
- Shows that even r=1 captures significant adaptation signal.

## Key Functions/Classes

| Name | Type | Purpose |
|------|------|---------|
| `LoRALinear` | nn.Module | Wraps frozen Linear + trainable low-rank A, B matrices |
| `SimpleTransformer` | nn.Module | Minimal Transformer encoder for classification |
| `apply_lora_to_model` | function | Replaces target Linear layers with LoRALinear |
| `count_trainable_parameters` | function | Counts params with `requires_grad=True` |
| `merge_lora_weights` | function | Merges BA into W₀ for zero-latency inference |

## Data Flow

```
Token IDs (batch, seq_len)
    → Embedding (batch, seq_len, d_model)
    → Positional Encoding
    → Transformer Encoder Layer × N
        → Multi-Head Attention
            → Wq, Wk, Wv, Wo projections [LoRA applied to Wq, Wv]
            → Scaled dot-product attention
        → Feedforward (2-layer MLP)
        → Residual + LayerNorm
    → Mean pooling (batch, d_model)
    → Classifier head (batch, num_classes)
    → Cross-entropy loss
```

## Deliberate Simplifications vs. Full Paper

1. **Model scale**: We use a tiny Transformer (d_model=64, 2 layers, 2 heads, vocab ~500) instead of GPT-3 175B or RoBERTa-large. The principle is identical.
2. **Pre-training**: We do a brief pre-training pass on the toy dataset rather than large-scale pre-training. The focus is on demonstrating the LoRA mechanism, not achieving SOTA.
3. **Task scope**: Single binary classification task instead of GLUE/WikiSQL/SAMSum benchmarks.
4. **LoRA targets**: We apply LoRA to Wq and Wv only (matching the paper's recommended configuration), but don't explore all combinations (Wq+Wk+Wv+Wo, MLP, etc.).
5. **No distributed training**: Single GPU/CPU, no model parallelism or gradient checkpointing.
6. **No quantization**: We use full fp32 weights; QLoRA and 8-bit variants are out of scope.
