# Code Architecture — Pix2Pix (Image-to-Image Translation with Conditional Adversarial Networks)

## Notebook Structure

### 1. Setup & Imports
- Install/import PyTorch, torchvision, matplotlib, numpy
- Set device (CUDA if available, else CPU)
- Define hyperparameters: image size (256×256), batch size, learning rate (0.0002), β1=0.5, λ_L1=100, epochs

### 2. Dataset Preparation
- **Edges2Shoes dataset** (from the Berkeley pix2pix repository): paired edge maps → shoe photos
- Download a small subset (~200 pairs) for demonstration
- Custom Dataset class: loads paired images, splits input (edges) and target (shoes), applies transforms (resize, normalize to [-1, 1], random jitter/mirror for augmentation)
- DataLoader with batch_size=1 (as in the paper for best results)

### 3. Generator Architecture (U-Net)
- **Encoder:** 8 downsampling blocks — each: Conv2d(stride=2) → BatchNorm2d → LeakyReLU(0.2)
  - Channels: 3→64→128→256→512→512→512→512→512 (last block no BatchNorm)
- **Decoder:** 8 upsampling blocks — each: ConvTranspose2d(stride=2) → BatchNorm2d → Dropout(0.5) → ReLU
  - Channels: 512→512→512→512→256→128→64→3
  - Skip connections: concatenate encoder layer i output with decoder layer n-i input
  - Final layer: Tanh activation (output in [-1, 1] range)
- Input: 3-channel edge image (256×256) → Output: 3-channel generated image (256×256)

### 4. Discriminator Architecture (PatchGAN)
- 70×70 PatchGAN: 4 convolutional layers
  - Conv2d(3+3→64, kernel=4, stride=2) → LeakyReLU(0.2)  [no BatchNorm]
  - Conv2d(64→128, kernel=4, stride=2) → BatchNorm → LeakyReLU(0.2)
  - Conv2d(128→256, kernel=4, stride=2) → BatchNorm → LeakyReLU(0.2)
  - Conv2d(256→512, kernel=4, stride=1) → BatchNorm → LeakyReLU(0.2)
  - Conv2d(512→1, kernel=4, stride=1) → Sigmoid
- Input: concatenation of input image + target (real) or input image + generated (fake)
- Output: N×N probability map (each value = probability that patch is real), averaged to scalar

### 5. Loss Functions
- **GAN Loss:** Binary Cross Entropy with Logits (BCEWithLogitsLoss)
  - D real loss: D(real_pair) should be 1
  - D fake loss: D(fake_pair) should be 0
  - G adversarial loss: D(fake_pair) should be 1 (fool D)
- **L1 Loss:** L1Loss between generated and target images, weighted by λ=100

### 6. Training Loop
- For each epoch, for each batch:
  1. Forward pass: G(input) → fake_output
  2. Train D: maximize log(D(real)) + log(1 - D(fake)), with .detach() on fake
  3. Train G: minimize -log(D(fake)) + λ * L1(fake, target)
  4. Use Adam optimizer for both G and D
  5. Apply dropout in G during both train and test
- Track losses for plotting

### 7. Visualization
- Side-by-side grid: Input (edges) | Generated | Target (real shoes)
- Loss curves for G and D over epochs
- Save sample outputs at intervals

## Key Functions/Classes

| Component | Description |
|-----------|-------------|
| `UNetGenerator` | U-Net with 8 encoder + 8 decoder blocks, skip connections, dropout on first 3 decoder blocks |
| `PatchGANDiscriminator` | 70×70 PatchGAN, 4 conv layers, outputs patch-level real/fake map |
| `Edge2ShoeDataset` | Custom Dataset: loads paired images, splits, augments |
| `train_epoch()` | One epoch: D step then G step, returns losses |
| `visualize_results()` | Grid of input/generated/target triples |

## Data Flow / Shapes

```
Input: edges (B, 3, 256, 256)
  → Generator (U-Net)
    → Encoder: 256→128→64→32→16→8→4→2→1 (spatial), 64→128→256→512 channels
    → Decoder: 1→2→4→8→16→32→64→128→256 (spatial) with skip concat
    → Output: (B, 3, 256, 256) in [-1, 1]
  
Discriminator input: cat(input, target_or_fake) → (B, 6, 256, 256)
  → 4 conv layers → (B, 1, 30, 30) patch map
  → mean → scalar probability
```

## Deliberate Simplifications vs Full Paper

1. **Dataset size:** Using ~200 paired images instead of the full edges2shoes dataset (~50k). Results will be lower quality but demonstrate the architecture.
2. **Training duration:** 200 epochs (reduced for Kaggle time limits) vs the paper's ~200 epochs on full data. We use a smaller dataset so convergence is faster.
3. **Noise injection:** Paper uses dropout at test time; we apply dropout during training only for simplicity and stability on a small dataset.
4. **No FCN-score evaluation:** We skip the FCN-8s semantic segmentation metric and use visual inspection only.
5. **No AMT perceptual study:** We don't run Amazon Mechanical Turk evaluations.
6. **Single task:** We demonstrate on edges→shoes only, whereas the paper tests on 8+ tasks.
7. **Image size:** 256×256 as in the paper — no change here.
