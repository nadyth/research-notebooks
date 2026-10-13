# Code Architecture — DDPM

## Overview

The notebook implements a simplified DDPM from scratch in PyTorch, training on a small image dataset (FashionMNIST or MNIST for speed) with a compact U-Net denoiser. The goal is to demonstrate the full pipeline: forward noising, U-Net training, and iterative sampling.

## Section-by-Section Breakdown

### 1. Setup & Hyperparameters
- **T = 200** (reduced from paper's 1000 for faster training)
- **betas:** linear schedule from 1e-4 to 0.02
- **Image size:** 32×32 (FashionMNIST resized) or 28×28
- **Batch size:** 128
- **Epochs:** 10-20 (reduced for Kaggle runtime budget)
- **Precompute:** alphas, alphas_cumprod (ᾱ_t), sqrt terms, and one-pass terms for the posterior q(x_{t-1}|x_t,x_0)

### 2. Forward Diffusion Process
- **`q_sample(x_0, t, noise)`:** Closed-form noising: x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε
- Precomputes all `sqrt_alphas_cumprod` and `sqrt_one_minus_alphas_cumprod` as buffers
- Demonstrates forward noising on sample images at various timesteps

### 3. U-Net Denoiser
Key components:
- **Sinusoidal time embedding:** Projects timestep t into a continuous embedding using sine/cosine at different frequencies, then through an MLP
- **Conv blocks:** Two conv layers + GroupNorm + SiLU (swish) activation
- **Down/Up path:** 3 downsample stages (channel dims: 64→128→256) with skip connections
- **Self-attention:** At the mid-block (lowest resolution) for global context
- **Time conditioning:** Time embedding added to each residual block via a linear projection
- Input: (B, C, H, W) + timestep → Output: (B, C, H, W) predicted noise

### 4. Training Loop
- For each batch: sample random timesteps t ~ Uniform({1,...,T}), sample noise ε ~ N(0,I)
- Compute x_t via q_sample
- Forward pass: ε_θ = model(x_t, t)
- Loss: MSE(ε, ε_θ) — the simplified training objective from the paper
- Optimizer: Adam, lr=2e-4
- No EMA for the simplified version

### 5. Sampling (Reverse Process)
- **`p_sample(model, x_t, t)`:** Single reverse step
  - Compute μ_θ = (1/√(ᾱ_t)) (x_t - (β_t/√(1-ᾱ_t)) ε_θ)
  - If t > 0: x_{t-1} = μ_θ + σ_t·z, where z ~ N(0,I), σ_t = √β_t
  - If t == 0: x_0 = μ_θ
- **`sample(model, shape)`:** Start from x_T ~ N(0,I), iterate t=T→0
- Generates and displays a grid of sampled images

### 6. Visualization
- Forward process demonstration: progressively noised images
- Training loss curve
- Final generated samples grid

## Data Flow / Shapes

```
Input image: (B, 1, 32, 32)  [FashionMNIST, single channel]
Noise ε:     (B, 1, 32, 32)
x_t:         (B, 1, 32, 32)  [noised image at timestep t]
U-Net output:(B, 1, 32, 32)  [predicted noise]
Loss:        scalar MSE

Sampling:
x_T:         (B, 1, 32, 32)  [pure noise]
→ iteratively denoise through T steps →
x_0:         (B, 1, 32, 32)  [generated image]
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|----------------|
| Timesteps T | 1000 | 200 (faster sampling) |
| Dataset | CIFAR-10 / LSUN-256 | FashionMNIST 32×32 (faster training) |
| Channels | 128 base, deep U-Net | 64 base, 3 stages (compact) |
| EMA | Yes, decay 0.9999 | No (simpler) |
| Attention | At 16×16, 8×8 | At mid-block only |
| Loss weighting | Weighted by 1/σ² | Simple unweighted MSE |
| Guidance | Unconditional | Unconditional |
| Training time | Hours/days on TPU/GPU | ~10 min on Kaggle GPU |

These simplifications preserve the core algorithm (forward noising, noise prediction, iterative reverse sampling) while fitting within Kaggle's 30-minute GPU limit.
