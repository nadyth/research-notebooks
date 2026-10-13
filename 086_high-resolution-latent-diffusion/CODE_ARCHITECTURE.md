# Code Architecture — Latent Diffusion Models (LDM)

## Overview

This notebook implements a miniature Latent Diffusion Model and compares it against a pixel-space DDPM on the same toy dataset. The pipeline has two stages: (1) train a small VAE to compress 32×32 images into an 8×8 latent space (f=4), then (2) train a U-Net denoiser in that latent space. We also train a pixel-space DDPM baseline for direct comparison of training/sampling speed and sample quality.

## Section-by-Section Breakdown

### 1. Setup and Imports
- PyTorch, torchvision (FashionMNIST/CIFAR-10 subset), matplotlib, numpy
- Device detection (cuda for Kaggle GPU)
- Utility: `show_images()` for grid visualization

### 2. Hyperparameters
- `IMG_SIZE=32`, `CHANNELS=1` (grayscale), `LATENT_CH=4`, `DOWNSAMPLE=4` (f=4)
- Latent spatial size: 32/4 = 8×8
- `T=200` diffusion timesteps
- `EPOCHS_VAE=5`, `EPOCHS_DIFFUSION=5`, `BATCH_SIZE=128`, `SUBSET_SIZE=8000`
- `BASE_CH=32` (compact U-Net)

### 3. Toy Dataset
- FashionMNIST subset (8000 images), 32×32 grayscale, normalized to [-1, 1]
- Single DataLoader shared by both VAE and diffusion training

### 4. Stage 1 — Autoencoder (Perceptual Compression)

**`VAE` class:**
- **Encoder:** 3 conv blocks (stride-2 downsampling) → 32→64→128 channels → 32×32 → 8×8 spatial
- Conv to `LATENT_CH=4` channels → latent shape (B, 4, 8, 8)
- **Decoder:** conv from latent → 3 upsample blocks (transposed conv) → 8→16→32 spatial → output (B, 1, 32, 32)
- **Loss:** reconstruction MSE + KL divergence (β=0.001 for mild regularization)
- Training: Adam, 5 epochs
- After training: **freeze** the VAE encoder/decoder

### 5. Forward Diffusion Process (in latent space)
- Linear beta schedule: β_t from 1e-4 to 0.02
- Precompute: alphas, alphas_cumprod (ᾱ_t), sqrt quantities
- `q_sample(z_0, t, noise)`: z_t = √ᾱ_t · z_0 + √(1-ᾱ_t) · ε
- **Key difference from pixel DDPM:** noise is added to latent codes (4×8×8) not pixels (1×32×32)

### 6. U-Net Denoiser (operates on latent space)
- **SinusoidalPositionEmbeddings**: timestep → sinusoidal vector
- **ResBlock**: Conv → GroupNorm → SiLU → Conv + time embedding injection + skip
- **Self-Attention**: at 4×4 latent resolution
- **U-Net**: down path (32→64→128) → bottleneck → up path with skip connections
- Input: (B, 4, 8, 8) + timestep → Output: (B, 4, 8, 8) predicted noise
- Architecture is identical to pixel DDPM U-Net but input/output channels = 4 (latent) instead of 1 (pixel)

### 7. Latent Diffusion Training
- Encode batch through frozen VAE encoder → z_0 (B, 4, 8, 8)
- Random timestep, forward diffusion on z_0, U-Net predicts noise, MSE loss
- Adam optimizer, 5 epochs
- Track training time for speed comparison

### 8. Pixel-Space DDPM Baseline
- Same U-Net architecture but input/output channels = 1 (pixel space)
- Same T=200, same dataset, same epochs
- Trained from scratch on 32×32 pixels for fair comparison
- Track training time

### 9. Sampling and Comparison

**`ddpm_sample(model, shape, T, ...)`** — standard DDPM reverse process:
- Start from pure noise → iteratively denoise through all T steps
- For LDM: sample in latent space → decode through frozen VAE decoder → image
- For pixel DDPM: sample directly in pixel space → image

**Speed comparison metrics:**
- Training time per epoch (LDM vs pixel DDPM)
- Sampling time for 16 images (LDM vs pixel DDPM)
- Latent diffusion should be ~f² = 16× faster per forward pass

**Quality comparison:**
- Visualize samples from both models side-by-side
- Report FID (if feasible) or visual quality assessment

### 10. Latent Space Analysis
- Visualize VAE reconstructions (encode → decode) to verify compression quality
- Show the latent representation's structure via interpolation
- Demonstrate that the autoencoder preserves perceptual quality despite 16× compression

## Data Flow / Shapes

```
Image: (B, 1, 32, 32)
  → VAE Encoder → Latent z: (B, 4, 8, 8)    [16× compression]
  → Diffusion on z: (B, 4, 8, 8) ↔ noise (B, 4, 8, 8)
  → U-Net operates on (B, 4, 8, 8)
  → Sampled z: (B, 4, 8, 8)
  → VAE Decoder → Generated Image: (B, 1, 32, 32)

Pixel DDPM baseline:
Image: (B, 1, 32, 32) ↔ noise (B, 1, 32, 32)
  → U-Net operates on (B, 1, 32, 32)
  → Sampled image: (B, 1, 32, 32)
```

## Deliberate Simplifications vs Full Paper

1. **No cross-attention conditioning:** The paper's main contribution for text-to-image is cross-attention layers in the U-Net. This notebook implements unconditional LDM for clarity. Cross-attention would require a text encoder and conditioning pipeline, adding complexity without illustrating the core latent-space insight.

2. **Small-scale VAE instead of VQGAN:** The paper uses a large VQGAN autoencoder trained on ImageNet. We train a tiny VAE on the same toy dataset. The principle (compress → diffuse → decompress) is identical.

3. **Grayscale 32×32 instead of RGB 256×256:** For Kaggle GPU time constraints. The paper works at 256²–1024² in RGB. The latent-space speedup scales with resolution, so the advantage is even more dramatic at higher resolutions.

4. **Linear noise schedule:** The paper discusses choosing the noise schedule based on the latent SNR. We use a simple linear schedule adapted to the latent's scale by normalizing the latent codes.

5. **No classifier-free guidance:** The paper uses guidance for conditional generation. Since we do unconditional generation, guidance is not applicable.

6. **Fewer timesteps (T=200 vs 1000):** For faster training/sampling within Kaggle's 30-minute limit. The principle holds at any T.
