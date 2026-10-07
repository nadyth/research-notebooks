# Code Architecture — CPC Notebook

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs/imports: `torch`, `torch.nn`, `numpy`, `matplotlib`, `scikit-learn` (for t-SNE/linear probe).
- Sets random seeds for reproducibility.
- Detects CUDA if available (Kaggle GPU), falls back to CPU.

### 2. Synthetic Sequential Dataset
- **Deliberate simplification:** Instead of LibriSpeech audio or ImageNet, we generate a synthetic sequential dataset with temporal structure.
- We create sequences of 2-D points drawn from one of K latent classes (e.g., 10 classes). Each class has a characteristic trajectory pattern (sine waves at different frequencies, linear ramps, etc.). The model sees only raw observations — no labels.
- Data shape: `(batch, seq_len, obs_dim)` where `obs_dim=2`, `seq_len=128`, `batch=64`.
- A `make_sequences()` function generates train/test splits.

### 3. Encoder Network (g_enc)
- A small 1-D convolutional network (2 conv layers + ReLU) that maps each observation x_t → latent z_t.
- Input: `(batch, seq_len, obs_dim)` → reshaped to `(batch, obs_dim, seq_len)` for Conv1d.
- Output: `(batch, seq_len, latent_dim)` where `latent_dim=64`.
- **Simplification:** The paper uses deep ResNet encoders; we use a 2-layer CNN.

### 4. Autoregressive Model (g_ar)
- A single-layer GRU that processes the encoder outputs z_{≤t} and produces context c_t at each step.
- Input: `(batch, seq_len, latent_dim)` → GRU → `(batch, seq_len, context_dim)` where `context_dim=64`.
- **Simplification:** Paper uses GRU for audio, PixelCNN for images; we use GRU for all.

### 5. InfoNCE Loss & Prediction Heads
- **W_k prediction matrices:** For each prediction step k (k ∈ {1, 2, 3, 5, 8}), a linear layer W_k maps c_t → predicted future latent ẑ_{t+k}.
- **InfoNCE computation:**
  - For a batch of B sequences, at time t and step k:
    - Positive sample: z_{t+k}^{(i)} — the true future latent of sequence i.
    - Negative samples: z_{t+k}^{(j)} for j ≠ i — futures of other sequences in the batch.
    - Score: f_k(x_{t+k}, c_t) = exp(z_{t+k} · W_k · c_t) — log-bilinear.
    - Loss: L_NCE = -log( exp(score_pos) / Σ_j exp(score_j) ) — cross-entropy over B candidates.
- **`info_nce_loss(context, future_latents, W_k)`:** vectorized implementation using batch matrix multiplication.
- **`CPCModel` class:** wraps encoder + GRU + prediction heads; `forward()` returns contexts and latents; `compute_loss()` aggregates InfoNCE across all prediction steps.

### 6. Training Loop
- Optimizer: Adam, lr=1e-3.
- Epochs: ~50 (enough for convergence on synthetic data).
- For each batch:
  1. Encode all observations → z_t.
  2. Run GRU → c_t.
  3. For each step k: compute InfoNCE loss using c_t and z_{t+k}.
  4. Sum losses across steps, backprop, step optimizer.
- Logs training loss, classification accuracy (accuracy of picking the correct positive), and mutual information estimate (log(B) - loss).

### 7. Evaluation — Linear Probe
- Freeze the CPC model. Extract context vectors c_t for all training sequences.
- Train a logistic regression (sklearn) on top of c_t to predict the ground-truth class label.
- Report linear probe accuracy on the test set.
- **Purpose:** Shows that CPC representations are linearly separable by class *without ever seeing labels during pretraining*.

### 8. Visualization
- **t-SNE plot:** t-SNE of context vectors c_t colored by true class label → shows learned representations cluster by class.
- **Accuracy vs. prediction step k:** Bar chart showing InfoNCE accuracy at each k (should decrease with larger k — farther futures are harder to predict).
- **Training loss curve:** InfoNCE loss over epochs.
- **MI estimate:** log(N) − L_N plotted over training.

## Key Functions/Classes

| Component | Signature | Purpose |
|---|---|---|
| `make_sequences()` | `(n_samples, seq_len, obs_dim, n_classes) → (X, y)` | Generate synthetic sequential data |
| `Encoder` | `nn.Module: (B, S, D_obs) → (B, S, D_lat)` | Conv1d-based observation encoder |
| `AutoregressiveModel` | `nn.Module: (B, S, D_lat) → (B, S, D_ctx)` | GRU-based context summarizer |
| `CPCModel` | `nn.Module` | Wraps encoder + AR + prediction heads |
| `info_nce_loss()` | `(c_t, z_future, W_k) → scalar` | Compute InfoNCE for one step |
| `compute_loss()` | `(contexts, latents, steps) → scalar` | Aggregate InfoNCE across all steps |
| `linear_probe()` | `(contexts, labels) → accuracy` | Train sklearn LogReg on frozen features |

## Data Flow / Shapes

```
Input:    (B=64, S=128, D_obs=2)
  → Encoder (Conv1d)
    z_t: (B=64, S=128, D_lat=64)
  → GRU
    c_t: (B=64, S=128, D_ctx=64)
  → W_k @ c_t  (for each k in {1,2,3,5,8})
    ẑ_{t+k}: (B=64, S=128, D_lat=64)
  → InfoNCE: compare ẑ_{t+k} vs z_{t+k} (positive) and z_{t+k} of other batch elements (negatives)
    Loss: scalar
```

## Deliberate Simplifications vs. Full Paper

| Aspect | Paper | Our Notebook |
|---|---|---|
| Data | LibriSpeech audio, ImageNet images, BookCorpus text | Synthetic 2-D sequential data with K classes |
| Encoder | 5-layer strided CNN (audio) / ResNet-101 (vision) | 2-layer Conv1d |
| Autoregressive model | GRU (audio) / PixelCNN (vision) / GRU (NLP) | Single-layer GRU |
| Prediction steps | Up to 12-20 steps | 5 steps {1,2,3,5,8} |
| Negative samples | N=256 (audio), varied strategies | Batch-size negatives (N=64) |
| Training | 8 GPUs, large batches | Single GPU, batch=64, 50 epochs |
| Downstream | Phone classification, ImageNet linear probe, 5 NLP tasks | Linear probe on synthetic classes |

These simplifications keep the notebook runnable within Kaggle's 30-minute limit while faithfully implementing the core algorithm: latent-space prediction + InfoNCE contrastive loss + mutual information maximization.
