# Code Architecture — Layer Normalization Notebook

## Overview

The notebook implements Layer Normalization from scratch in NumPy and PyTorch, plugs it into an RNN/Transformer block, and compares training stability against BatchNorm on sequence data. All computation is CPU-friendly (no GPU required).

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `numpy`, `matplotlib` for visualization
- Random seed fixing for reproducibility

### 2. LayerNorm from Scratch in NumPy
- **Function `layer_norm_numpy(x, gain, bias, eps=1e-5)`**
  - Input: `x` is a 1D array of H elements (the summed inputs to a layer for one sample)
  - Compute mean μ = mean(x) across all H elements
  - Compute variance σ² = var(x) across all H elements
  - Normalize: x_norm = (x - μ) / sqrt(σ² + eps)
  - Scale and shift: out = gain * x_norm + bias
  - Returns normalized output
- **Gradient verification**: numerical gradient check against analytic gradients
  - Verify dL/dx, dL/dgain, dL/dbias via finite differences
  - Assert relative error < 1e-5

### 3. LayerNorm as a PyTorch Module
- **Class `LayerNorm(nn.Module)`**
  - `__init__(features, eps=1e-5)`: initializes `self.gamma = nn.Parameter(torch.ones(features))` and `self.beta = nn.Parameter(torch.zeros(features))`
  - `forward(x)`: x has shape (batch, seq_len, features) or (batch, features)
    - Compute mean and variance over the **last dimension** (features)
    - Normalize and apply gain/bias
  - Key shapes:
    - Input: (B, H) → mean/var over dim=-1, keepdim=True
    - Input: (B, T, H) → mean/var over dim=-1, keepdim=True
    - Output: same shape as input

### 4. BatchNorm for Comparison
- **Class `BatchNorm1d(nn.Module)`** — simplified from-scratch version
  - Normalizes over the batch dimension (dim=0)
  - Maintains running mean/var for inference
  - `forward(x)`: during training, use batch statistics and update running stats; during eval, use running stats

### 5. Toy Sequence Classification Task
- Generate a synthetic sequence dataset:
  - Each sample is a sequence of T=20 timesteps with H=64 features
  - Two classes: "increasing trend + noise" vs "decreasing trend + noise"
  - 1000 train samples, 200 test samples
  - Data shapes: X_train (1000, 20, 64), y_train (1000,)

### 6. Simple RNN with Normalization
- **Class `RNNCellWithNorm(nn.Module)`**
  - Computes: a_t = W_hh @ h_{t-1} + W_xh @ x_t
  - Applies normalization (LayerNorm or BatchNorm or None) to a_t
  - Applies non-linearity: h_t = tanh(a_t_normalized)
  - Parameters: W_hh (H, H), W_xh (H, H), plus norm module
- **Class `SimpleRNN(nn.Module)`**
  - Wraps the RNN cell, unrolls over T timesteps
  - Final hidden state → linear classifier → class logits
  - Forward: iterates over timesteps, applies cell, returns logits from final h

### 7. Training Loop
- Three models: NoNorm, BatchNorm, LayerNorm — identical architecture except normalization
- Optimizer: Adam, lr=0.001
- Loss: CrossEntropyLoss
- Epochs: 50
- Record training loss, test accuracy per epoch
- Key data flow:
  ```
  X (B,T,H) → RNN unroll → h_T (B,H) → Linear(H, 2) → logits (B,2) → CrossEntropy
  ```

### 8. Transformer Block with LayerNorm
- **Class `TransformerBlock(nn.Module)`**
  - Self-attention (simplified, single-head) + LayerNorm + feedforward + LayerNorm
  - Pre-norm style: LayerNorm before attention and before FFN
  - Demonstrates where LayerNorm sits in a transformer block
  - Apply to the same sequence classification task

### 9. Visualization & Comparison
- Plot 1: Training loss curves (NoNorm vs BatchNorm vs LayerNorm)
- Plot 2: Test accuracy curves over epochs
- Plot 3: Hidden state magnitude over timesteps (showing LayerNorm stabilizes dynamics)
- Plot 4: Gradient flow comparison (gradient norm per layer over training)

### 10. Analysis
- Print final accuracies for all three normalization schemes
- Discuss: LayerNorm converges faster, stabilizes hidden state, doesn't depend on batch size
- Show that BatchNorm with batch_size=1 degrades (online learning scenario) while LayerNorm doesn't

## Key Functions/Classes
| Component | Purpose |
|---|---|
| `layer_norm_numpy()` | Pure NumPy implementation for gradient verification |
| `LayerNorm(nn.Module)` | Reusable PyTorch LayerNorm module |
| `BatchNorm1d(nn.Module)` | From-scratch BatchNorm for comparison |
| `RNNCellWithNorm` | RNN cell with pluggable normalization |
| `SimpleRNN` | Full RNN model for sequence classification |
| `TransformerBlock` | Simplified transformer block with LayerNorm |

## Data Flow & Shapes
```
Input: (B=64, T=20, H=64)
  ↓ RNN unroll (t=1..20)
  h_t: (64, 64) at each step
  ↓ final h_T
  h_T: (64, 64)
  ↓ Linear(64, 2)
  logits: (64, 2)
  ↓ CrossEntropy
  loss: scalar
```

## Deliberate Simplifications vs Full Paper
1. **Toy dataset** — synthetic trend classification instead of real NLP tasks (MNIST, COCO, CNN corpus)
2. **Small model** — 64 hidden units, 20 timesteps (paper used 2400-dim encoders)
3. **No convolutional experiments** — paper has a ConvNet section; we focus on RNN/Transformer which is where LayerNorm shines
4. **Single-head attention** — simplified transformer block instead of multi-head
5. **No skip-thought / order-embedding experiments** — these require large datasets and multi-day training
6. **Gradient analysis is empirical** — paper's Riemannian geometry analysis is theoretical; we show practical gradient flow instead
7. **Batch size sensitivity** — demonstrated empirically rather than through the paper's invariance proofs
