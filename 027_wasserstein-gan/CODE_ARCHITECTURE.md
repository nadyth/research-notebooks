# Code Architecture — Wasserstein GAN Notebook

## Overview

This notebook implements Wasserstein GAN from scratch using PyTorch, training on a toy 2D distribution (a 2D Gaussian ring/mixture) to demonstrate the key properties of the WGAN algorithm: stable critic loss that correlates with sample quality, and the effect of weight clipping to enforce the Lipschitz constraint.

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `torch.optim` for model and training
- `numpy` for data generation
- `matplotlib.pyplot` for visualization
- Set random seeds for reproducibility
- Detect CUDA device

### 2. Toy Data Distribution
- **Real distribution:** 2D Gaussian mixture arranged in a ring (8 modes around a circle)
- This is a classic toy problem where vanilla GANs often suffer from mode collapse
- `sample_real(n)` returns n samples from the ring distribution
- Visualize the real distribution with a scatter plot

### 3. Generator Network
- **Architecture:** 2-layer MLP (latent_dim=2 → hidden=128 → output_dim=2)
- Input: noise vector z ~ N(0, I) of dimension 2
- Output: 2D point (x, y)
- LeakyReLU activations, final layer has Tanh (to bound output range)
- Simple enough to train fast, expressive enough to learn the ring

### 4. Critic Network (Discriminator without sigmoid)
- **Architecture:** 2-layer MLP (input_dim=2 → hidden=128 → 1)
- Input: 2D point
- Output: scalar score (real-valued, NO sigmoid)
- LeakyReLU activations
- This is the key difference from vanilla GAN: outputs a real-valued function f(x), not a probability

### 5. WGAN Training Loop
- **Hyperparameters:**
  - `n_critic = 5` (critic updates per generator update)
  - `lr = 5e-5` (learning rate, low as specified in the paper)
  - `clip_value = 0.01` (weight clipping bound)
  - `n_iterations = 3000` (total generator iterations)
  - `batch_size = 256`
- **Critic update step:**
  1. Sample real batch and noise
  2. Generate fake batch
  3. Compute critic loss = mean(critic(fake)) - mean(critic(real))
  4. Backprop and step optimizer (maximize critic score on real, minimize on fake)
  5. **Clamp all critic parameters to [-clip_value, clip_value]** ← this enforces Lipschitz
- **Generator update step (every n_critic iterations):**
  1. Sample noise
  2. Generate fake batch
  3. Compute generator loss = -mean(critic(fake))
  4. Backprop and step optimizer

### 6. Monitoring & Visualization
- Record critic loss and generator loss at each generator step
- Plot the loss curves to show the critic loss is meaningful (decreasing, correlating with quality)
- At several checkpoints, plot generated samples vs real samples as scatter plots
- Show how generated distribution evolves from random blob → ring shape

### 7. Analysis: WGAN vs Vanilla GAN Loss Behavior
- Compare the WGAN critic loss with a vanilla GAN discriminator loss on the same setup
- Show that vanilla GAN loss saturates (stays near 0 or log(2)) while WGAN loss provides a smooth gradient signal
- Plot the correlation between critic loss and a sample quality metric (e.g., mean distance of generated samples to nearest real mode)

## Key Functions/Classes

| Component | Description |
|-----------|-------------|
| `Generator(nn.Module)` | 2-layer MLP mapping latent z → 2D point |
| `Critic(nn.Module)` | 2-layer MLP mapping 2D point → scalar score |
| `sample_real(n)` | Samples n points from the 2D ring distribution |
| `train_wgan()` | Main training loop implementing the WGAN algorithm |
| `plot_samples()` | Scatter plot of real vs generated samples |
| `plot_losses()` | Plot critic and generator loss curves |

## Data Flow / Shapes

```
Noise z: [batch, 2]  →  Generator  →  fake data: [batch, 2]
Real data: [batch, 2]  →  Critic  →  real scores: [batch, 1]
Fake data: [batch, 2]  →  Critic  →  fake scores: [batch, 1]
Critic loss = mean(fake_scores) - mean(real_scores)  →  scalar
Generator loss = -mean(fake_scores)  →  scalar
```

## Deliberate Simplifications vs Full Paper

1. **Toy 2D distribution instead of images:** The paper uses LSUN Bedrooms and CIFAR-10. We use a 2D Gaussian ring for fast training and clear visualization. The WGAN algorithm is identical.
2. **Small MLPs instead of DCGAN architecture:** The paper uses DCGAN-style convolutional networks. We use simple 2-layer MLPs suitable for 2D data.
3. **RMSProp replaced with Adam:** The original paper recommends RMSProp; we use Adam with low lr (a common modern practice that works equally well for toy problems).
4. **No inception score / FID evaluation:** We use visual inspection and loss correlation instead, which is more appropriate for 2D toy data.
5. **Fixed weight clipping:** We use the original weight clipping method (c=0.01) rather than the gradient penalty from WGAN-GP, since the purpose is to demonstrate the original WGAN algorithm.
