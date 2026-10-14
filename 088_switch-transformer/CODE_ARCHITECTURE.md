# Code Architecture — Switch Transformer Notebook

## Section-by-Section Breakdown

### 1. Imports and Configuration
- PyTorch, torch.nn, math, time, matplotlib
- Hyperparameters: `d_model=128`, `num_heads=4`, `d_ff=512`, `num_experts=8`, `num_layers=2`, `vocab_size=1000`, `seq_len=64`, `batch_size=64`, `epochs=20`, `capacity_factor=1.25`, `aux_loss_weight=0.01`

### 2. Toy Dataset: Synthetic Token Sequence Task
- Generates synthetic sequences of random token IDs with a simple rule-based label (e.g., parity of token sum mod num_classes).
- This avoids needing large NLP datasets while still demonstrating routing behavior.
- Data shapes: `(batch_size, seq_len)` input, `(batch_size,)` target labels.
- A simple token embedding layer maps token IDs to `(batch_size, seq_len, d_model)`.

### 3. Dense FFN Module (Baseline)
- Standard two-layer FFN with ReLU/GELU: `Linear(d_model, d_ff)` → `GELU` → `Linear(d_ff, d_model)`.
- Residual connection + LayerNorm around the FFN.
- This is the baseline that every token passes through.

### 4. Switch Router
- **Class `SwitchRouter(nn.Module)`**: A single linear layer `W_r: (d_model, num_experts)` that maps each token to expert logits.
- Forward: `logits = x @ W_r.T` → `probs = softmax(logits, dim=-1)` → `expert_idx = argmax(probs, dim=-1)`.
- Returns: `expert_indices (batch, seq)`, `router_probs (batch, seq, num_experts)`.
- Router z-loss: `z_loss = mean(sum(exp(logits), dim=-1))` — penalizes large pre-softmax logits for stability.

### 5. Switch MoE Layer (Core)
- **Class `SwitchMoELayer(nn.Module)`**: Contains `num_experts` independent FFN modules + a `SwitchRouter`.
- **Forward pass:**
  1. Compute router indices and probabilities.
  2. For each expert, gather tokens routed to it (using `expert_indices`).
  3. Apply capacity factor: if more tokens are routed to an expert than its capacity, drop excess tokens (they pass through via residual).
  4. Each expert processes its assigned tokens through its FFN.
  5. Scatter outputs back to original positions.
  6. Scale expert outputs by the router probability for that expert (for differentiability).
  7. Compute auxiliary load-balancing loss: `L_aux = alpha * N * sum(f_i * P_i)`.
- Data flow: `(B, S, D)` → router → per-expert gather → expert FFNs → scatter back → `(B, S, D)`.
- Key shapes:
  - Input: `(batch, seq, d_model)`
  - Router logits: `(batch, seq, num_experts)`
  - Expert indices: `(batch, seq)` — each value in `[0, num_experts)`
  - Per-expert token tensors: variable length, up to `expert_capacity = ceil(batch * seq / num_experts * capacity_factor)`

### 6. Transformer Encoder Layer
- **Class `TransformerEncoderLayer(nn.Module)`**: Multi-head self-attention → LayerNorm → (Switch MoE or Dense FFN) → LayerNorm.
- Two variants: one with dense FFN, one with Switch MoE layer.
- Both share the same attention mechanism for fair comparison.

### 7. Model Assembly
- **Class `SwitchTransformer(nn.Module)`**: Embedding → N × encoder layer → mean pooling → classifier head.
- **Class `DenseTransformer(nn.Module)`**: Same structure but with dense FFN layers.
- Both report parameter counts for comparison.

### 8. Training Loop
- Trains both models with same optimizer (AdamW), same learning rate, same data.
- Records loss, accuracy, and wall-clock time per epoch.
- The Switch model's total loss = task loss + aux_loss + z_loss.

### 9. Evaluation and Comparison
- Plots: (a) training loss curves, (b) accuracy curves, (c) wall-clock time per epoch.
- Table: parameter count, final accuracy, total training time, samples/second.
- Expert utilization histogram: shows how many tokens each expert received (to verify load balancing).

### 10. Analysis: Expert Routing Visualization
- For a sample batch, shows which expert each token was routed to.
- Verifies that the load-balancing loss achieves roughly uniform expert usage.

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `SwitchRouter` | Routes tokens to experts via top-1 softmax selection |
| `SwitchMoELayer` | Contains N expert FFNs, dispatches tokens, computes aux loss |
| `TransformerEncoderLayer` | Attention + MoE/dense FFN + LayerNorm |
| `SwitchTransformer` | Full model: embedding + layers + classifier |
| `DenseTransformer` | Baseline: same architecture with dense FFN |
| `compute_aux_loss()` | Load-balancing loss: `α·N·Σ(f_i·P_i)` |
| `train_model()` | Training loop with loss/time tracking |
| `plot_comparison()` | Side-by-side comparison plots |

## Data Flow / Shapes

```
Input tokens (B, S)
    ↓ Embedding
(B, S, D)
    ↓ Multi-Head Attention
(B, S, D)
    ↓ SwitchRouter
router_probs (B, S, E), expert_idx (B, S)
    ↓ Gather per-expert tokens
E × (≤capacity, D)
    ↓ Expert FFNs (parallel)
E × (≤capacity, D)
    ↓ Scatter back
(B, S, D)
    ↓ Mean pool + Classifier
(B, num_classes)
```

Where B=batch, S=seq_len, D=d_model, E=num_experts.

## Deliberate Simplifications vs. Full Paper

1. **Scale:** 8 experts × 128 d_model vs. 128 experts × 768 d_model (T5-Base scale).
2. **Task:** Synthetic sequence classification, not masked language modeling on C4.
3. **No distributed training:** Single device, no all-to-all communication across devices.
4. **No bfloat16 / mixed precision:** Full FP32 training.
5. **No distillation:** We only compare sparse vs. dense, not distill sparse → dense.
6. **No expert dropout / noise:** Paper's more advanced stability tricks omitted for clarity.
7. **Fixed capacity factor:** Paper sweeps this; we fix at 1.25.
