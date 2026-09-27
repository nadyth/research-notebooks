# Code Architecture — Improved Training of Wasserstein GANs (WGAN-GP) Notebook

## Overview

This notebook implements WGAN-GP from scratch using PyTorch, training on a toy 2D distribution (the same 2D Gaussian ring used in the WGAN notebook) to demonstrate the key improvement: the gradient penalty replaces weight clipping for enforcing the Lipschitz constraint. The notebook then compares training stability curves between WGAN (weight clipping) and WGAN-GP (gradient penalty) on the same toy dataset.

## Section-by-Section Breakdown

### 1. Imports & Setup
- `torch`, `torch.nn`, `torch.optim` for model and training
- `torch.autograd.grad` for computing the gradient penalty (requires per-sample gradients)
- `numpy` for data generation
- `matplotlib.pyplot` for visualization
- Set random seeds for reproducibility
- Detect CUDA device

### 2. Toy Data Distribution
- **Real distribution:** 2D Gaussian mixture arranged in a ring (8 modes around a circle)
- Same distribution as the WGAN notebook for fair comparison
- `sample_real(n)` returns n samples from the ring distribution
- Visualize the real distribution with a scatter plot

### 3. Generator Network
- **Architecture:** 2-layer MLP (latent_dim=2 → hidden=128 → output_dim=2)
- Input: noise vector z ~ N(0, I) of dimension 2
- Output: 2D point (x, y)
- LeakyReLU activations, final layer has Tanh (to bound output range)
- Identical to the WGAN generator for fair comparison

### 4. Critic Network (no weight clipping, no batch norm)
- **Architecture:** 2-layer MLP (input_dim=2 → hidden=128 → 1)
- Input: 2D point
- Output: scalar score (real-valued, NO sigmoid)
- LeakyReLU activations
- **No weight clipping** — weights are unconstrained
- **No BatchNorm** — would interfere with per-sample gradient computation for the penalty. LayerNorm is used instead (or no normalization for this small MLP)

### 5. Gradient Penalty Function
- **Key function:** `gradient_penalty(critic, real, fake)`
  1. Sample epsilon ~ U[0, 1] of shape [batch, 1]
  2. Compute interpolations: `interpolated = epsilon * real + (1 - epsilon) * fake`
  3. Make interpolated require gradients: `interpolated.requires_grad_(True)`
  4. Compute critic output on interpolated: `critic_interpolated = critic(interpolated)`
  5. Compute gradients: `gradients = autograd.grad(outputs=critic_interpolated, inputs=interpolated, grad_outputs=torch.ones_like(critic_interpolated), create_graph=True, retain_graph=True)`
  6. Reshape gradients to [batch, -1] and compute L2 norm: `gradient_norm = gradients.view(batch, -1).norm(2, dim=1)`
  7. Penalty = `lambda * ((gradient_norm - 1) ** 2).mean()`
- **`create_graph=True`** is critical: we need to backpropagate through the gradient computation itself (second-order gradients)
- **`lambda = 10`** as recommended in the paper

### 6. WGAN-GP Training Loop
- **Hyperparameters:**
  - `n_critic = 5` (critic updates per generator update)
  - `lr = 1e-4` (Adam, higher than WGAN's 5e-5 since no weight clipping constrains the optimization)
  - `beta1 = 0.5, beta2 = 0.9` (Adam betas, as recommended in the paper)
  - `lambda_gp = 10` (gradient penalty weight)
  - `n_iterations = 3000` (total generator iterations)
  - `batch_size = 256`
- **Critic update step:**
  1. Sample real batch and noise
  2. Generate fake batch
  3. Compute critic loss = `mean(C(fake)) - mean(C(real))`
  4. Compute gradient penalty on interpolations
  5. Total critic loss = critic loss + gradient penalty
  6. Backprop and step optimizer
  7. **NO weight clipping** ← this is the key change from WGAN
- **Generator update step (every n_critic iterations):**
  1. Sample noise
  2. Generate fake batch
  3. Compute generator loss = `-mean(C(fake))`
  4. Backprop and step optimizer

### 7. WGAN (Weight Clipping) Training Loop
- Identical setup but with weight clipping (clip_value=0.01) and no gradient penalty
- Uses RMSProp with lr=5e-5 (as in original WGAN)
- Trained for the same number of iterations for fair comparison

### 8. Results — Stability Curve Comparison
- **Loss curves:** Plot critic loss for both WGAN and WGAN-GP over training
  - WGAN-GP loss should be smoother and more stable
  - WGAN loss may show oscillation or instability, especially with suboptimal clip values
- **Gradient norm analysis:** Track the critic's gradient norm at interpolation points
  - WGAN-GP: gradient norm stays close to 1 (enforced by penalty)
  - WGAN: gradient norm can vary wildly (only crudely bounded by weight clipping)
- **Sample quality evolution:** Plot generated samples at checkpoints for both methods
  - Both should learn the ring, but WGAN-GP typically covers all 8 modes more reliably

### 9. Analysis — Gradient Penalty Behavior
- Plot the gradient penalty term over training to show it converges (critic becomes approximately 1-Lipschitz)
- Compare the critic weight distribution: WGAN weights are clamped to [-0.01, 0.01], WGAN-GP weights are unconstrained
- Show that WGAN-GP allows the critic to use its full capacity while maintaining the Lipschitz property

### 10. Key Takeaways
1. Gradient penalty > weight clipping for enforcing Lipschitz constraint
2. No hyperparameter tuning needed for the clipping threshold
3. WGAN-GP enables higher learning rates and Adam optimizer
4. Gradient penalty is a soft constraint — it encourages but doesn't strictly enforce 1-Lipschitz
5. The interpolation strategy (sampling on the real-fake line) is optimal because it targets the transport path

## Key Functions/Classes

| Component | Description |
|-----------|-------------|
| `Generator(nn.Module)` | 2-layer MLP mapping latent z → 2D point |
| `Critic(nn.Module)` | 2-layer MLP mapping 2D point → scalar score (no weight clipping) |
| `sample_real(n)` | Samples n points from the 2D ring distribution |
| `gradient_penalty(critic, real, fake)` | Computes the WGAN-GP gradient penalty term |
| `train_wgan_gp()` | Main training loop for WGAN-GP |
| `train_wgan()` | Main training loop for WGAN (weight clipping) for comparison |
| `plot_comparison()` | Side-by-side comparison of WGAN vs WGAN-GP results |

## Data Flow / Shapes

```
Noise z: [batch, 2]  →  Generator  →  fake data: [batch, 2]
Real data: [batch, 2]  →  Critic  →  real scores: [batch, 1]
Fake data: [batch, 2]  →  Critic  →  fake scores: [batch, 1]
Interpolated: [batch, 2]  →  Critic  →  interp scores: [batch, 1]
                                ↓ autograd.grad
Gradients: [batch, 2]  →  L2 norm  →  [batch]  →  penalty: scalar

Critic loss = mean(fake_scores) - mean(real_scores) + lambda * penalty  →  scalar
Generator loss = -mean(fake_scores)  →  scalar
```

## Deliberate Simplifications vs Full Paper

1. **Toy 2D distribution instead of images:** The paper uses LSUN Bedrooms, CIFAR-10, and CelebA. We use a 2D Gaussian ring for fast training and clear visualization. The WGAN-GP algorithm is identical.
2. **Small MLPs instead of DCGAN/ResNet architectures:** The paper demonstrates on 101-layer ResNets. We use simple 2-layer MLPs suitable for 2D data.
3. **No Inception Score / FID evaluation:** We use visual inspection and loss stability analysis instead, which is more appropriate for 2D toy data.
4. **No language model experiment:** The paper includes a discrete data (character-level language model) experiment. We focus on the continuous case where the gradient penalty is most directly applicable.
5. **Fixed lambda=10:** The paper found lambda=10 to be robust across all settings; we use this fixed value rather than doing a sweep.
6. **LayerNorm instead of no normalization:** For the small MLP, we use no normalization (the network is simple enough). The paper uses LayerNorm for the critic (instead of BatchNorm) in larger architectures because BN interferes with the gradient penalty.
