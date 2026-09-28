# Code Architecture — Progressive Growing of GANs

## Overview

The notebook implements a simplified Progressive Growing GAN (ProGAN) in PyTorch, following the key architectural and training innovations from Karras et al. (2017). The implementation grows both generator and discriminator from 4×4 to 64×64 resolution, with smooth fade-in transitions between resolution stages.

## Section-by-Section Breakdown

### 1. Configuration & Hyperparameters
- **Latent dimension:** 128 (input noise vector size)
- **Channel progression:** 512 → 512 → 256 → 128 → 64 (decreasing as resolution increases)
- **Resolutions:** 4×4 → 8×8 → 16×16 → 32×32 → 64×64
- **Fade-in iterations:** 200 per stage transition
- **Stabilization iterations:** 300 per stage (after fade-in completes)
- **Batch sizes:** vary by resolution (larger at low res, smaller at high res)
- **LeakyReLU slope:** 0.2 (as per paper)

### 2. Equalized Learning Rate Convolution (`EqualizedConv2d`, `EqualizedConvTranspose2d`)
- Custom wrapper around `nn.Conv2d` / `nn.ConvTranspose2d`
- Weights initialized from N(0, 1), then scaled at runtime by `c = √(2 / fan_in)`
- The `fan_in` accounts for kernel size, input channels, and spatial receptive field
- Deliberate simplification: the paper uses a slightly different gain calculation; here we use the standard He variance scaling for simplicity

### 3. Pixelwise Feature Normalization (`PixelNorm`)
- Applied after each conv layer in the generator (not the discriminator)
- Formula: `x = x / sqrt(mean(x², dim=1, keepdim=True) + ε)` where ε = 1e-8
- Normalizes per-pixel across the channel dimension

### 4. Minibatch Standard Deviation (`MinibatchStdDev`)
- Appended as the first layer of the discriminator's final block
- Computes per-spatial-location stddev across the batch, averages it, and appends as one extra channel
- Implementation: compute std → reduce to mean → expand as (B, 1, H, W) → concat with input

### 5. Generator Architecture (`Generator`)
- **Initial block:** Linear(latent_dim, 4×4×512) → reshape → LReLU → toRGB(1×1 conv)
- **Growth blocks:** each adds an Upsample(2×) → Conv3×3 → LReLU → Conv3×3 → LReLU
- **toRGB:** 1×1 conv to 3 channels at each resolution (shared per stage)
- **Fade-in:** during transition, the output is blended:
  `output = (1 - α) * upsample(toRGB(prev_features)) + α * toRGB(new_features)`
- α linearly goes from 0 → 1 during fade-in iterations

### 6. Discriminator Architecture (`Discriminator`)
- **fromRGB:** 1×1 conv from 3 channels at each resolution
- **Growth blocks (mirror):** Conv3×3 → LReLU → Conv3×3 → LReLU → AvgPool(2×) (downsampling)
- **Final block at 4×4:** MinibatchStdDev → Conv3×3 → LReLU → Flatten → Linear → 1
- **Fade-in:** during transition:
  `input = (1 - α) * AvgPool(real_image) + α * fromRGB(downsampled_new_features)`

### 7. Training Loop
- **Resolution schedule:** for each resolution stage, run fade-in iterations then stabilization iterations
- **n_critic:** discriminator is updated once per generator update (1:1 ratio, as in the paper)
- **Loss:** non-saturating GAN loss (BCE with logits)
- **Optimizer:** Adam with lr=0.001, beta1=0.0, beta2=0.99 (as per paper)
- **Data:** MNIST images upscaled to current resolution (simplification — paper uses CelebA)
- **Visualization:** saves generated samples at the end of each resolution stage

### 8. Visualization
- Grid of generated samples at each resolution (4×4, 8×8, 16×16, 32×32, 64×64)
- Uses torchvision utils `make_grid` for clean image grids
- Final plot shows the progression of image quality across resolution stages

## Key Functions/Classes

| Class/Function | Purpose |
|---|---|
| `EqualizedConv2d` | Conv layer with runtime weight scaling for equalized learning rate |
| `EqualizedConvTranspose2d` | Transpose conv with equalized learning rate (for generator upsampling) |
| `PixelNorm` | Per-pixel L2 normalization across channels |
| `MinibatchStdDev` | Minibatch stddev layer for mode-collapse detection |
| `GeneratorBlock` | Single generator growth block (upsample + 2 convs + toRGB) |
| `DiscriminatorBlock` | Single discriminator growth block (2 convs + downsample + fromRGB) |
| `Generator` | Full progressive generator with grow() and forward(alpha) methods |
| `Discriminator` | Full progressive discriminator with grow() and forward(alpha) methods |
| `train_step` | Single D + G update step at a given resolution and alpha |
| `grow_and_train` | Outer loop: for each resolution, fade-in then stabilize |

## Data Flow / Shapes

```
Latent z (B, 128) 
  → Linear → (B, 512, 4, 4) 
  → [Growth: Upsample ×2 → Conv → LReLU → Conv → LReLU]
  → 8×8 → 16×16 → 32×32 → 64×64
  → toRGB (1×1 conv) → (B, 3, H, W)

Real image (B, 3, H, W)
  → fromRGB (1×1 conv)
  → [Growth: Conv → LReLU → Conv → LReLU → AvgPool ×2]
  → 32×32 → 16×16 → 8×8 → 4×4
  → MinibatchStdDev → Conv3×3 → LReLU → Flatten → Linear → (B, 1)
```

## Deliberate Simplifications vs Full Paper

1. **Dataset:** Uses MNIST (upscaled) instead of CelebA/HQ. The paper trains on CelebA HQ at 1024×1024; this notebook uses MNIST at up to 64×64 for time efficiency on Kaggle.
2. **Max resolution:** 64×64 instead of 1024×1024. Each doubling requires significantly more compute.
3. **Equalized learning rate:** Uses He-based scaling `√(2/fan_in)` rather than the exact formula from the paper which uses `√(2) * √(1/fan_in)` — functionally equivalent.
4. **Upsampling:** Uses `nn.Upsample` (nearest neighbor) + conv instead of transposed convolutions. Both are valid; the paper uses transposed conv but nearest + conv is more stable.
5. **EMA (Exponential Moving Average):** The paper uses EMA of generator weights for final sample generation. Omitted here for simplicity.
6. **Truncation trick:** Not implemented (a post-training inference trick from the paper).
7. **Loss:** Standard non-saturating GAN loss. The paper uses the same, but also experiments with WGAN-GP loss.
8. **Iteration counts:** Greatly reduced from the paper's millions of iterations to ~500 per stage for practical runtime on Kaggle.
