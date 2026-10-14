# Code Architecture — Sparsely-Gated Mixture-of-Experts

## Overview

The notebook implements a self-contained Mixture-of-Experts (MoE) layer with Noisy Top-K Gating in PyTorch, trains it on a character-level language modeling task, and visualizes expert activation patterns.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, `matplotlib`, `collections`
- Sets random seed for reproducibility
- Auto-detects CUDA device

### 2. Data Preparation
- Downloads a small text corpus (Shakespeare or similar public-domain text)
- Creates character-level vocabulary (character → index mapping)
- Generates sequences of fixed length for training
- Creates PyTorch DataLoader with batched sequences
- **Shapes**: input `(batch, seq_len)`, target `(batch, seq_len)` shifted by 1

### 3. Expert Network (`Expert` class)
- Simple feed-forward network: Linear(input_dim, hidden_dim) → ReLU → Linear(hidden_dim, output_dim)
- Each expert has identical architecture but independent parameters
- **Shapes**: input `(batch, seq_len, d_model)` → output `(batch, seq_len, d_model)`

### 4. Noisy Top-K Gating (`NoisyTopKGate` class)
**Key function**: the gating network that routes inputs to experts.

- `W_g`: trainable weight matrix for gating logits `(d_model, num_experts)`
- `W_noise`: trainable weight matrix for noise scaling `(d_model, num_experts)`
- Forward pass:
  1. Compute clean logits: `logits = x @ W_g` → shape `(batch*seq_len, num_experts)`
  2. Compute noise: `noise = StandardNormal() * Softplus(x @ W_noise)`
  3. Add noise to logits: `noisy_logits = logits + noise`
  4. Keep top-k values, set rest to −∞: `KeepTopK(noisy_logits, k)`
  5. Apply softmax to get gating weights: `G = Softmax(topk_logits)`
  6. Return gating weights `G` and indices of selected experts
- **Shapes**: input `(batch, seq_len, d_model)` → gates `(batch*seq_len, num_experts)` (sparse, only k non-zero per row), top_k_indices `(batch*seq_len, k)`

### 5. MoE Layer (`MoELayer` class)
- Combines gating network + list of expert networks
- Forward pass:
  1. Flatten input across batch and sequence dimensions
  2. Compute gating weights and selected expert indices
  3. For each expert, gather the inputs routed to it, compute expert output
  4. Multiply expert outputs by gating weights and scatter back
  5. Reshape back to `(batch, seq_len, d_model)`
- Also returns gating weights for loss computation and visualization
- **Deliberate simplification**: processes all experts in a loop (not batched across experts) for clarity — the paper uses distributed model parallelism across GPUs

### 6. Load Balancing Losses
- **Importance loss**: `CV(Σ_batch G(x))²` — coefficient of variation squared of per-expert gate value sums across the batch. Encourages equal total gate weight per expert.
- **Load loss**: accounts for the probability that each expert is in the top-k considering the noisy gating. Computed via the CDF of the Gaussian noise distribution.
- Total MoE auxiliary loss: `w_importance * L_importance + w_load * L_load`

### 7. Language Model with MoE (`MoELanguageModel` class)
- Architecture: `Embedding → LSTM → MoE → LSTM → Linear(output)`
- The MoE layer sits between two LSTM layers (as in the paper's Figure 1)
- Forward returns logits and MoE auxiliary losses
- **Simplification**: small-scale (128-dim, 16 experts, k=2) vs paper's 4-billion parameter models with thousands of experts

### 8. Training Loop
- Cross-entropy loss for next-character prediction
- Total loss = task loss + MoE auxiliary losses (importance + load)
- Adam optimizer, configurable epochs (kept small for Kaggle runtime)
- Prints loss and expert utilization stats every N steps
- **Simplification**: single GPU, no distributed training (paper uses 16-32 GPUs with data+model parallelism)

### 9. Expert Activation Visualization
- After training, feeds different input patterns through the model
- Records which experts are selected for each input
- **Visualization 1**: heatmap of expert activation frequency (which experts fire most often)
- **Visualization 2**: per-input expert selection patterns (showing different inputs route to different experts)
- **Visualization 3**: training progression of expert utilization balance over epochs

### 10. Analysis & Conclusions
- Checks expert specialization: do certain experts prefer certain character types?
- Prints load balance statistics
- Compares MoE model vs equivalent-cost dense model (no MoE, same active parameter count)

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `Expert` | Single feed-forward expert sub-network |
| `NoisyTopKGate` | Trainable sparse gating with noise + top-k selection |
| `MoELayer` | Combines gate + experts, routes inputs, returns outputs + aux losses |
| `MoELanguageModel` | Full model: Embedding → LSTM → MoE → LSTM → Output |
| `importance_loss` | Encourages equal total gate weight per expert |
| `load_loss` | Encourages balanced example counts per expert |

## Data Flow / Shapes

```
Input text → char indices (B, L) 
  → Embedding (B, L, d_model) 
  → LSTM1 (B, L, d_model) 
  → MoE: 
    → Gate (B*L, num_experts) → top-k selection (B*L, k)
    → Expert outputs (B*L, k, d_model) → weighted sum (B, L, d_model)
  → LSTM2 (B, L, d_model) 
  → Linear (B, L, vocab_size) 
  → Loss (cross-entropy)
```

## Deliberate Simplifications vs Full Paper

1. **Scale**: 16 experts × ~50K params each vs paper's 4096 experts × 1M params each (137B total)
2. **No distributed training**: single GPU loop over experts vs paper's data+model parallelism across 16-32 GPUs
3. **Character-level LM** vs paper's word-level 1-Billion-Word benchmark
4. **No hierarchical MoE**: flat single-level gating vs paper's hierarchical MoE for large expert counts
5. **No batching optimization**: the "shrinking batch problem" and convolutional batching techniques are discussed but not implemented (irrelevant at this scale)
6. **No CNN/convolutional application**: paper applies MoE convolutionally across time steps; we flatten batch+seq for simplicity
