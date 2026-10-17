# Code Architecture: Chinchilla Scaling Laws Notebook

## Overview

The notebook implements a simplified but complete reproduction of the Chinchilla scaling law analysis. It trains several small character-level transformer language models at different parameter counts and training-token combinations, records the final cross-entropy loss for each, and fits the parametric scaling law L(N,D) = E + A/N^α + B/D^β to the collected data. It then derives the compute-optimal allocation N_opt(C) and D_opt(C), compares it to the Kaplan et al. (2020) allocation, and visualizes the scaling-law frontier — exactly as the paper does with its 400+ models, but at a miniature scale runnable on a single GPU within 30 minutes.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import PyTorch, numpy, scipy (for curve fitting), matplotlib.
- Set random seeds for reproducibility.
- Configure device (GPU/CPU).
- Define hyperparameter grid: list of model configurations (n_layer, n_embd, n_head) and token counts to sweep.

### 2. Toy Text Corpus
- Generate or load a small text corpus (Shakespeare-like or synthetic structured text).
- Build a character-level tokenizer (vocab ~65 characters).
- **Data shapes**: Input `(batch, seq_len)`, target `(batch, seq_len)` — standard next-token prediction.
- The corpus is large enough to train on varying numbers of tokens without repetition issues.

### 3. Small Transformer Language Model
- **Key class**: `MiniTransformer(nn.Module)` — a minimal GPT-style decoder with:
  - Token + positional embeddings
  - N Transformer decoder blocks (causal multi-head self-attention + feedforward + residual + layer norm)
  - LM head (tied weights optional)
- Configurable `n_embd` (embedding dim), `n_layer` (depth), `n_head` (attention heads), `vocab_size`, `max_seq_len=256`.
- Model sizes in the sweep: ~10K, ~50K, ~100K, ~200K, ~500K parameters (small enough to train many variants quickly).
- **Forward shapes**: `(batch, seq_len)` → embeddings `(batch, seq_len, n_embd)` → blocks `(batch, seq_len, n_embd)` → logits `(batch, seq_len, vocab_size)`.
- **Parameter counting**: A helper `count_parameters(model)` returns total non-embedding params, matching the paper's convention (N excludes embedding/token tables where possible).

### 4. Training Sweep — Multiple (N, D) Combinations
- **Key function**: `train_model(config, n_tokens, ...)` — trains one model with the given config for n_tokens tokens and returns the final validation loss.
- Sweep over a grid of model sizes × token counts:
  - 5 model sizes (varying n_embd and n_layer)
  - 4-5 token counts per model (e.g., 1M, 2M, 5M, 10M, 20M tokens)
  - Total: ~20-25 training runs
- Each run: AdamW optimizer, cosine LR schedule, ~short training (just enough to reach the loss curve's smooth region for that token count).
- Record: (N_params, D_tokens, final_loss, compute_flops) for each run.
- **Compute**: FLOPs ≈ 6 * N * D (forward + backward, as in the paper).
- **Data shapes**: Each training step processes `(batch_size, seq_len)` where `batch_size * seq_len = tokens_per_step`.

### 5. Fitting the Parametric Scaling Law
- **Key function**: `fit_scaling_law(loss_data)` — uses `scipy.optimize.curve_fit` to fit:
  ```
  L(N, D) = E + A / N^alpha + B / D^beta
  ```
  where (E, A, B, alpha, beta) are free parameters fitted to the (N, D, loss) data.
- Initial guesses: E~1.0, A~100, B~100, alpha~0.3, beta~0.3 (based on paper's values).
- Report fitted parameters and compare to paper's values (E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28).
- **Loss landscape visualization**: 3D or contour plot of L(N, D) surface with fitted parameters.

### 6. Deriving Compute-Optimal Allocation
- **Key function**: `compute_optimal_ND(C, E, A, B, alpha, beta)` — given compute budget C, minimize L(N,D) subject to 6*N*D = C.
  - Using Lagrange multipliers or numerical optimization (scipy.optimize.minimize with constraint).
  - Derive: N_opt(C) ∝ C^(beta/(alpha+beta)), D_opt(C) ∝ C^(alpha/(alpha+beta)).
  - For alpha=beta: N_opt ∝ C^0.5, D_opt ∝ C^0.5 (equal scaling — the Chinchilla result).
- Plot N_opt and D_opt as functions of C on a log-log scale.
- Compare to Kaplan et al.'s prediction: N_opt ∝ C^0.73, D_opt ∝ C^0.27.

### 7. IsoFLOP Analysis (Approach 2)
- For a fixed compute budget C, select several model sizes and train each for the token count D = C / (6*N).
- Plot final loss vs model size for each IsoFLOP curve.
- Find the minimum of each curve → optimal N for that C.
- Repeat for 3-4 compute levels and show the optimal N scales as C^0.5.

### 8. Visualization & Comparison
- **Plot 1**: Loss vs FLOPs for all training runs, colored by model size. Show the compute-optimal frontier (envelope of minimum-loss points).
- **Plot 2**: Log-log plot of N_opt(C) and D_opt(C) vs C, with fitted power laws. Overlay Kaplan et al.'s prediction for contrast.
- **Plot 3**: Parametric loss surface L(N, D) with the compute-optimal curve overlaid.
- **Plot 4**: Bar chart comparing Chinchilla allocation (equal N/D scaling) vs Kaplan allocation (N-heavy) at a representative compute budget.
- **Summary table**: Fitted parameters vs paper's parameters, and predicted optimal N/D ratio.

## Key Functions/Classes

| Component | Class/Function | Purpose |
|---|---|---|
| Tokenizer | `CharTokenizer` | Character-level tokenizer for the toy corpus |
| Language Model | `MiniTransformer` | GPT-style decoder for language modeling |
| Param Counter | `count_parameters()` | Count non-embedding model parameters |
| Training | `train_model()` | Train one model for N tokens, return final loss |
| Sweep | `run_sweep()` | Train models across the (N, D) grid |
| Scaling Law Fit | `fit_scaling_law()` | Fit L = E + A/N^α + B/D^β via scipy curve_fit |
| Optimal Allocation | `compute_optimal_ND()` | Derive N_opt, D_opt for a given compute budget C |
| IsoFLOP Analysis | `run_isoflop()` | Train at fixed FLOPs, varying model size |
| Visualization | `plot_results()` | Generate all comparison plots |

## Data Flow

```
Toy Corpus → CharTokenizer → Training Grid (N × D)
                                ↓
                        train_model() for each (N, D)
                                ↓
                    (N, D, loss, FLOPs) records
                                ↓
                    ┌───────────┴───────────┐
                    ↓                       ↓
            fit_scaling_law()         run_isoflop()
            (parametric fit)          (Approach 2)
                    ↓                       ↓
            E, A, B, α, β            N_opt per FLOP level
                    ↓                       ↓
            compute_optimal_ND()     compare approaches
                    ↓                       ↓
                    └───────────┬───────────┘
                                ↓
                    plot_results()
                    (frontier, N_opt vs C, Kaplan comparison)
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|---|---|---|
| Model sizes | 70M to 16B+ parameters | ~10K to ~500K parameters |
| Training data | 5B to 500B tokens (MassiveText) | 1M to 20M tokens (toy corpus) |
| Number of models | 400+ training runs | ~20-25 training runs |
| Tokenizer | SentencePiece/byte-level BPE | Character-level (~65 vocab) |
| Architecture | Full GPT-style transformer (Gopher-like) | Mini GPT (2-6 layers, 64-256 embd) |
| Compute budget | ~10^21 to 10^25 FLOPs | ~10^8 to 10^11 FLOPs |
| FLOP formula | 6*N*D (same) | 6*N*D (same) |
| Scaling law params | E=1.69, A=406.4, B=410.7, α=0.34, β=0.28 | Fitted from toy data (values will differ) |
| IsoFLOP curves | Trained at 9 FLOP levels | 3-4 FLOP levels |
| Chinchilla model | 70B params, 1.4T tokens | Not trained (too large); inferred from scaling law |
| Evaluation | MMLU, BIG-bench, etc. | Validation loss only |

These simplifications keep the notebook runnable within Kaggle's 30-minute GPU limit while faithfully implementing all three scaling-law estimation approaches and the parametric loss fitting from the paper.
