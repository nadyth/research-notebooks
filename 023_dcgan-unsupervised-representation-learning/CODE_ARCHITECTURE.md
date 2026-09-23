# Code Architecture — DCGAN Notebook

## Overview

This notebook implements the DCGAN architecture from Radford et al. (2015) following the paper's architectural guidelines. A convolutional generator and discriminator are trained adversarially on a small image dataset (Fashion-MNIST upscaled to 64×64), and the latent space is visualized through interpolation walks.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, torchvision, matplotlib, numpy
- Sets random seeds for reproducibility
- Detects CUDA device

### 2. Hyperparameters
- `latent_dim = 100` — dimension of the noise vector z (matches the paper)
- `img_size = 64` — output image resolution (the paper used 64×64 for LSUN)
- `channels = 1` — grayscale (Fashion-MNIST); the paper used 3-channel RGB
- `ngf = 64` — number of generator feature maps (base multiplier)
- `ndf = 64` — number of discriminator feature maps
- `lr = 0.0002` — learning rate (paper found 0.001 too high)
- `beta1 = 0.5` — Adam β₁ (reduced from default 0.9 for stability)
- `batch_size = 128` — matches the paper
- `num_epochs = 50` — reduced from paper's longer training for Kaggle time limits
- `leaky_slope = 0.2` — LeakyReLU slope in discriminator

### 3. Data Loading
- Loads Fashion-MNIST via torchvision
- Resizes images to 64×64 (required for the DCGAN architecture's 4 conv layers)
- Normalizes to [-1, 1] range to match the generator's Tanh output
- Creates DataLoader with batch_size=128

**Data flow:** Fashion-MNIST 28×28 → Resize(64×64) → Normalize([-1,1]) → tensor [B, 1, 64, 64]

**Simplification:** The paper trained on LSUN bedrooms (3M RGB images), ImageNet-1k, and a faces dataset. We use Fashion-MNIST (grayscale, 60K images) for reproducibility and speed on Kaggle.

### 4. Generator Network (G)

Architecture (input → output shapes):
1. **FC + reshape:** z [B, 100] → [B, ngf*8, 4, 4] (project and reshape to 4×4 spatial)
2. **ConvTranspose2d:** ngf*8 → ngf*4, kernel 4, stride 2, pad 1 → [B, 256, 8, 8] + BatchNorm + ReLU
3. **ConvTranspose2d:** ngf*4 → ngf*2, kernel 4, stride 2, pad 1 → [B, 128, 16, 16] + BatchNorm + ReLU
4. **ConvTranspose2d:** ngf*2 → ngf, kernel 4, stride 2, pad 1 → [B, 64, 32, 32] + BatchNorm + ReLU
5. **ConvTranspose2d:** ngf → channels, kernel 4, stride 2, pad 1 → [B, 1, 64, 64] + Tanh

**Key design choices per paper:**
- No BatchNorm on the final output layer (paper: "not applying batchnorm to the generator output layer")
- ReLU everywhere except output (Tanh)
- Fractional-strided convolutions (ConvTranspose2d) for upsampling — no pooling
- No fully connected layers after the initial projection

**Weight init:** Custom function initializes ConvTranspose2d weights from N(0, 0.02) and BatchNorm weights from N(1.0, 0.02), biases to 0.

### 5. Discriminator Network (D)

Architecture (input → output shapes):
1. **Conv2d:** channels → ndf, kernel 4, stride 2, pad 1 → [B, 64, 32, 32] + LeakyReLU(0.2) (no BatchNorm on first layer)
2. **Conv2d:** ndf → ndf*2, kernel 4, stride 2, pad 1 → [B, 128, 16, 16] + BatchNorm + LeakyReLU(0.2)
3. **Conv2d:** ndf*2 → ndf*4, kernel 4, stride 2, pad 1 → [B, 256, 8, 8] + BatchNorm + LeakyReLU(0.2)
4. **Conv2d:** ndf*4 → ndf*8, kernel 4, stride 2, pad 1 → [B, 512, 4, 4] + BatchNorm + LeakyReLU(0.2)
5. **Conv2d:** ndf*8 → 1, kernel 4, stride 1, pad 0 → [B, 1, 1, 1] → flatten → Sigmoid

**Key design choices per paper:**
- No BatchNorm on the input layer (paper: "not applying batchnorm to the discriminator input layer")
- LeakyReLU with slope 0.2 throughout
- Strided convolutions for downsampling — no pooling
- Final layer is a single sigmoid output (no fully connected hidden layers)

### 6. Loss Function & Optimizers
- **BCE Loss** (Binary Cross Entropy) — the standard GAN loss
- Labels: real = 1, fake = 0
- Two separate Adam optimizers (G and D), both with lr=0.0002, betas=(0.5, 0.999)

**Simplification:** The paper used the original GAN minimax loss. We use the BCE formulation which is mathematically equivalent but more numerically stable to implement.

### 7. Training Loop

For each epoch, for each batch:
1. **Train Discriminator:**
   - Forward real images through D → loss with label 1
   - Generate fake images with G → forward through D → loss with label 0
   - Backprop total D loss, step optimizer
2. **Train Generator:**
   - Generate fake images → forward through D → loss with label 1 (want to fool D)
   - Backprop G loss, step optimizer
3. **Record losses** for plotting
4. **Save generated samples** from fixed noise every few epochs for animation

**Fixed noise:** A fixed batch of 64 noise vectors is used throughout training to visualize how the same latent points evolve.

### 8. Loss Curves
- Plots generator and discriminator losses over training iterations
- Shows the characteristic adversarial oscillation

### 9. Latent Space Interpolation Walk
- Picks two random noise vectors z₁ and z₂
- Linearly interpolates between them in N steps
- Generates images at each interpolation point
- Displays as a row showing smooth transitions — the "walking in the latent space" experiment from Section 6.1 of the paper

### 10. Generated Sample Grid
- Shows a grid of generated images from the trained generator
- Demonstrates the diversity and quality of outputs

### 11. Real vs Fake Comparison
- Side-by-side display of real Fashion-MNIST images and generated images
- Visual qualitative assessment

### 12. Summary
- Recap of what was built
- Key takeaways and comparison to the paper's findings

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Dataset | LSUN bedrooms (3M RGB), ImageNet-1k, faces (350K) | Fashion-MNIST (60K grayscale) |
| Image size | 64×64 RGB | 64×64 grayscale |
| Training duration | 5+ epochs on 3M images (~15M images seen) | 50 epochs on 60K images (~3M images seen) |
| Feature extraction eval | CIFAR-10 + SVHN with L2-SVM | Not included (focus on generation) |
| Vector arithmetic | Face arithmetic (glasses, gender) | Latent interpolation walk only |
| Guided backprop visualization | Discriminator feature viz | Not included |
| Forgetting objects | Window feature dropout | Not included |

The simplifications keep the notebook runnable within Kaggle's 30-minute GPU limit while demonstrating the core DCGAN architecture and training procedure.
