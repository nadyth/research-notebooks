# Code Architecture — DDIM

## Overview

This notebook implements DDIM on top of a DDPM-trained U-Net denoiser. The key difference from the DDPM notebook (Sr 84) is entirely in the **sampling** step: DDIM replaces the stochastic, step-by-step reverse Markov chain with a deterministic, subsequence-based sampler. The training procedure and U-Net architecture are identical.

## Section-by-Section Breakdown

### 1. Setup and Imports
- PyTorch, torchvision (FashionMNIST), matplotlib, numpy
- Device detection (cuda for Kaggle GPU)

### 2. Hyperparameters
- `T=100` total diffusion timesteps (same as DDPM)
- `IMG_SIZE=32`, `CHANNELS=1` (grayscale FashionMNIST)
- `EPOCHS=3`, `BATCH_SIZE=64`, `SUBSET_SIZE=5000`
- `BASE_CH=32` (compact U-Net for fast training)
- DDIM-specific: `DDIM_STEPS=20` (sample using only 20 of 100 timesteps)

### 3. Forward Diffusion Process
- Linear beta schedule: β_t from 1e-4 to 0.02
- Precompute: alphas, alphas_cumprod (ᾱ_t), sqrt quantities
- `q_sample(x_0, t, noise)`: closed-form noising x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε

### 4. Dataset and DataLoader
- FashionMNIST subset (5000 images), 32×32, normalized to [-1, 1]

### 5. U-Net Denoiser (identical to DDPM)
- **SinusoidalPositionEmbeddings**: timestep → sinusoidal vector
- **ResBlock**: Conv → GroupNorm → SiLU → Conv + time embedding injection + skip
- **Self-Attention**: at 16×16 resolution
- **U-Net**: down path (32→64→128) → bottleneck → up path with skip connections
- Input: (B, 1, 32, 32) + timestep → Output: (B, 1, 32, 32) predicted noise

### 6. Training Loop
- Simplified objective: L = E[||ε - ε_θ(x_t, t)||²]
- Random timestep sampling, forward diffusion, noise prediction, MSE loss
- Adam optimizer, 3 epochs

### 7. DDIM Sampling (THE KEY INNOVATION)

**`ddim_sample(model, x_T, ddim_steps, eta=0.0)`**

1. **Subsequence selection:** Choose S evenly-spaced timesteps from [0, T):
   `τ_i = (i · T) // S` for i = S, S-1, ..., 1

2. **For each step i (from S down to 1):**
   - t_current = τ_i, t_prev = τ_{i-1} (or 0 if i=1)
   - Predict noise: ε_θ(x_t, t_current)
   - Predict x_0: x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
   - Compute direction: dir = √(1-ᾱ_{t_prev}) · ε_θ
   - Update: x_{t_prev} = √ᾱ_{t_prev} · x̂_0 + dir
   - If η > 0: add stochastic noise scaled by σ_t

3. **eta parameter:** η=0 → pure DDIM (deterministic), η=1 → equivalent to DDPM

### 8. DDPM Sampling (for comparison)
- Standard stochastic p_sample loop through all T=100 steps
- Used as baseline to compare against DDIM's 20-step sampling

### 9. Comparison: DDIM vs DDPM
- Generate samples with both methods from the same initial noise
- Compare visual quality and wall-clock time
- DDIM with 20 steps should be ~5× faster with comparable quality

### 10. Latent Interpolation
- Generate two images from two random x_T codes
- Spherical interpolate (slerp) between the two x_T vectors
- Generate images at interpolation points — semantic morphing

## Key Functions

| Function | Input | Output | Purpose |
|---|---|---|---|
| `q_sample(x_0, t, noise)` | (B,C,32,32), int, (B,C,32,32) | (B,C,32,32) | Forward noising |
| `ddim_sample(model, x_T, steps, eta)` | model, (B,C,32,32), int, float | (B,C,32,32) | Fast deterministic sampling |
| `p_sample_loop(model, x_T)` | model, (B,C,32,32) | (B,C,32,32) | Standard DDPM sampling |
| `slerp(t, x0, x1)` | float, tensors | tensor | Spherical interpolation |

## Data Flow / Shapes

```
x_0: (B, 1, 32, 32)  →  q_sample  →  x_t: (B, 1, 32, 32)
                                           ↓
                                    U-Net(x_t, t)
                                           ↓
                                    ε_θ: (B, 1, 32, 32)
                                           ↓
                              DDIM update (20 steps)
                                           ↓
                                    x_0: (B, 1, 32, 32)  [generated image]
```

## Deliberate Simplifications vs Full Paper

1. **T=100 instead of 1000:** The paper uses T=1000; we use 100 for faster training. DDIM's speedup is still demonstrated (20 vs 100 steps).
2. **Linear schedule only:** The paper also explores cosine schedules; we use linear for simplicity.
3. **No classifier-free guidance:** The paper focuses on unconditional generation.
4. **FashionMNIST instead of CIFAR-10/ImageNet:** Smaller dataset for fast training within GPU time limits.
5. **Compact U-Net:** 32-base-channel U-Net instead of the larger architectures in the paper.
6. **eta=0 only for main sampling:** We demonstrate deterministic DDIM; the η>0 stochastic generalization is implemented but not the focus.
