# Code Architecture — StyleGAN Notebook

## Overview

This notebook implements a **miniature StyleGAN** that captures the paper's core architectural innovations: a mapping network, AdaIN-based style injection, learned constant input, stochastic noise injection, and style mixing. The implementation is deliberately scaled down to run within Kaggle's 30-minute GPU limit on a toy dataset (MNIST or small synthetic face-like data).

## Section-by-Section Breakdown

### 1. Imports and Configuration
- PyTorch, torchvision, matplotlib, numpy
- Hyperparameters: latent_dim=256, w_dim=256, image_size=32, channels=16, lr=1e-3, epochs=150
- Device selection (CUDA if available)

### 2. Toy Dataset
- Uses MNIST resized to 32×32 (or a small synthetic colored shape dataset)
- Simple, fast to train, available without download on Kaggle
- DataLoader with batch_size=64

### 3. Mapping Network (`MappingNetwork`)
```
Input: z (B, 256) ~ N(0, I)
Architecture: 8 × [Linear(256, 256) → LeakyReLU(0.2)]
Output: w (B, 256)
```
- 8-layer MLP as in the paper
- LeakyReLU with α=0.2
- Deliberate simplification: 256-dim instead of 512-dim to reduce parameters

### 4. AdaIN Module (`AdaIN`)
```
Input: x (B, C, H, W), y_style (B, 2*C)
  y_s = y_style[:, :C]  -- scale
  y_b = y_style[:, C:]  -- bias
Process:
  x_norm = (x - mean(x, dim=[2,3])) / std(x, dim=[2,3])   # per-channel instance norm
  output = y_s[:, :, None, None] * x_norm + y_b[:, :, None, None]
```
- Implements Eq. 1 from the paper exactly
- Per-channel normalization then affine transform with style

### 5. Styled Conv Block (`StyledConvBlock`)
```
Per block:
  1. Conv2d(in_ch, out_ch, 3, padding=1) with equalized LR
  2. Add noise: x + noise_weight * randn(B, 1, H, W)
  3. LeakyReLU(0.2)
  4. AdaIN(style from w)
  5. Conv2d(out_ch, out_ch, 3, padding=1)
  6. Add noise
  7. LeakyReLU(0.2)
  8. AdaIN(style from w)
```
- Two convolutions per resolution (matching paper's 2 layers per resolution)
- Noise is single-channel, broadcast via learned per-channel scaling
- Style affine transform: `A(w) = Linear(w_dim, 2*channels)` — learned affine

### 6. Synthesis Network (`SynthesisNetwork`)
```
Start: learned constant (1, channels, 4, 4) — nn.Parameter
Layers:
  4×4:   StyledConvBlock(ch, ch) → upsample
  8×8:   StyledConvBlock(ch, ch) → upsample
  16×16: StyledConvBlock(ch, ch) → upsample
  32×32: StyledConvBlock(ch, ch) → toRGB
toRGB: Conv2d(ch, 3, 1) -- 1×1 conv to 3-channel RGB
```
- Starts from learned constant (not random input)
- Progressive upsampling via bilinear interpolation
- 1×1 conv to RGB at the end
- Deliberate simplification: 4 resolutions (4²→32²) instead of 9 (4²→1024²)

### 7. Generator (`StyleGenerator`)
```
Forward(z):
  w = mapping_network(z)         # z → w
  img = synthesis_network(w)     # w → image
  return img
```
- Combines mapping + synthesis
- No progressive growing fade-in (simplified)

### 8. Discriminator (`Discriminator`)
```
Input: image (B, 3, 32, 32)
Architecture: 
  Conv2d(3, ch, 3, 2, 1) → LeakyReLU
  Conv2d(ch, ch*2, 3, 2, 1) → LeakyReLU
  Conv2d(ch*2, ch*4, 3, 2, 1) → LeakyReLU
  Conv2d(ch*4, 1, 4) → squeeze
Output: logits (B, 1)
```
- Standard DCGAN-style discriminator
- Not modified from baseline (paper doesn't change discriminator)

### 9. Equalized Learning Rate
- Weights initialized from N(0, 1), scaled by constant at runtime
- `weight = weight * (fan_in ** -0.5)` — equalized LR trick from Progressive GAN

### 10. Training Loop
- Loss: Non-saturating GAN loss (hinge loss variant)
- R1 regularization on discriminator (γ=10)
- Separate learning rates: mapping network gets 0.01× base LR
- Adam optimizer (β1=0, β2=0.99)
- Mixing regularization: 50% of batches use two latent codes with random crossover
- Exponential moving average of generator weights

### 11. Style Mixing Visualization
- Generate two images from z1, z2
- Mix styles: use w1 for coarse layers, w2 for fine layers
- Display grid showing: source A, source B, and mixed results at different crossover points
- Demonstrates the scale-specific control of styles

### 12. Stochastic Variation Visualization
- Fix w, vary noise inputs
- Generate multiple images with same style but different noise
- Show that high-level structure is preserved while details (texture, edges) vary

## Key Data Flow / Shapes

```
z: (B, 256) 
  → MappingNetwork → w: (B, 256)
  → SynthesisNetwork:
      const: (B, 16, 4, 4) 
      → StyledConv(4²) → upsample → (B, 16, 8, 8)
      → StyledConv(8²) → upsample → (B, 16, 16, 16)
      → StyledConv(16²) → upsample → (B, 16, 32, 32)
      → StyledConv(32²) → toRGB → (B, 3, 32, 32)
  → output: (B, 3, 32, 32)
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Image resolution | 1024×1024 | 32×32 |
| Latent dim | 512 | 256 |
| Feature maps | 512→16 (progressive) | 16 (fixed) |
| Mapping network depth | 8 layers | 8 layers (same) |
| Synthesis layers | 18 (2 per resolution, 9 resolutions) | 8 (2 per resolution, 4 resolutions) |
| Dataset | FFHQ (70K faces) | MNIST (or synthetic shapes) |
| Training time | ~1 week on 8×V100 | ~5-10 min on 1×T4 |
| Progressive growing | Yes (with fade-in) | No (fixed resolution) |
| Truncation trick | In 𝒲 space | Not implemented (visualization only) |
| Perceptual path length metric | Yes | Not computed |
| Linear separability metric | Yes | Not computed |

These simplifications are necessary to fit within Kaggle's 30-minute GPU limit while still demonstrating all core architectural components: mapping network, AdaIN style injection, learned constant, noise injection, and style mixing.
