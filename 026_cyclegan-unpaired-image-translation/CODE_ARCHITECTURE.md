# Code Architecture — CycleGAN Notebook

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install/import PyTorch, torchvision, matplotlib, numpy, PIL.
- Set device to CUDA if available.
- Define hyperparameters: image size (128×128), batch size, learning rate, lambda_cycle, lambda_identity, epochs.

### 2. Dataset Preparation
- Create/load two unpaired image domains (X and Y).
- Uses synthetic data generation (e.g., geometric shapes with different styles/colors) or downloads a small public dataset.
- Custom Dataset class that samples from domain X and domain Y independently (no pairing).
- Transforms: resize, random crop, normalize to [-1, 1].

### 3. Generator Architecture (ResNetGenerator)
```
Input (3×128×128)
  → ConvBlock(3, 64, k7, s1, p3) + InstanceNorm + ReLU    # c7s1-64
  → ConvBlock(64, 128, k3, s2, p1) + IN + ReLU            # d128
  → ConvBlock(128, 256, k3, s2, p1) + IN + ReLU           # d256
  → ResidualBlock(256) × 4                                 # R256 (simplified: 4 instead of 9)
  → ConvTransposeBlock(256, 128, k3, s2, p1, op1) + IN + ReLU  # u128
  → ConvTransposeBlock(128, 64, k3, s2, p1, op1) + IN + ReLU   # u64
  → ConvBlock(64, 3, k7, s1, p3) + Tanh                    # c7s1-3
Output (3×128×128)
```
Key classes:
- `ResidualBlock`: two Conv2d + InstanceNorm + ReLU with skip connection.
- `ConvBlock`: Conv2d + (optional) InstanceNorm + activation.
- `ConvTransposeBlock`: ConvTranspose2d + InstanceNorm + ReLU.

### 4. Discriminator Architecture (PatchGAN / NLayerDiscriminator)
```
Input (3×128×128)
  → Conv2d(3, 64, k4, s2, p1) + LeakyReLU(0.2)            # no norm on first layer
  → Conv2d(64, 128, k4, s2, p1) + IN + LeakyReLU(0.2)
  → Conv2d(128, 256, k4, s2, p1) + IN + LeakyReLU(0.2)
  → Conv2d(256, 1, k4, s1, p1)                              # output: 1×N×N patch map
Output (1×30×30 for 128×128 input)
```
- Classifies each patch as real/fake. Output is a grid of values (not a single scalar).
- Loss is MSE on real (all ones) / fake (all zeros) patch maps.

### 5. Loss Functions
- **Adversarial loss (LSGAN):** MSE between D(real) and 1, D(fake) and 0.
- **Cycle consistency loss:** L1(F(G(x)) - x) + L1(G(F(y)) - y), weighted by λ_cycle=10.
- **Identity loss:** L1(G(y) - y) for G and L1(F(x) - x) for F, weighted by λ_identity=0.5.
  - This encourages the generator to preserve color when the input is already in the target domain.

### 6. Training Loop
For each epoch:
1. Sample batch from domain X (real_A) and domain Y (real_B).
2. **Generators:**
   - fake_B = G(real_A), rec_A = F(fake_B)
   - fake_A = F(real_B), rec_B = G(fake_A)
   - identity_B = G(real_B), identity_A = F(real_A)
   - G_loss = LSGAN(D_Y(fake_B), 1) + λ_cycle * L1(rec_A, real_A) + λ_identity * L1(identity_B, real_B)
   - F_loss = LSGAN(D_X(fake_A), 1) + λ_cycle * L1(rec_B, real_B) + λ_identity * L1(identity_A, real_A)
   - Backprop total generator loss.
3. **Discriminators:**
   - D_Y_loss = LSGAN(D_Y(real_B), 1) + LSGAN(D_Y(fake_B.detach()), 0)
   - D_X_loss = LSGAN(D_X(real_A), 1) + LSGAN(D_X(fake_A.detach()), 0)
   - Backprop each discriminator loss separately.
4. Log losses, periodically save generated samples.

### 7. Visualization & Output
- Show side-by-side: real_A → fake_B → reconstructed_A.
- Show side-by-side: real_B → fake_A → reconstructed_B.
- Plot loss curves for G, F, D_X, D_Y.

## Data Flow / Tensor Shapes
| Stage | Shape (batch=1, 128×128) |
|---|---|
| real_A / real_B | (1, 3, 128, 128) |
| fake_B = G(real_A) | (1, 3, 128, 128) |
| D_Y(fake_B) | (1, 1, 30, 30) |
| rec_A = F(fake_B) | (1, 3, 128, 128) |

## Deliberate Simplifications vs Full Paper
1. **4 residual blocks** instead of 9 (reduces model size and training time).
2. **128×128 resolution** instead of 256×256.
3. **Synthetic/small dataset** instead of full horse2zebra or cityscapes.
4. **Fewer epochs (20-30)** instead of 100-200.
5. **No learning rate decay** (constant lr for simplicity).
6. **Smaller batch size** to fit Kaggle GPU memory.
