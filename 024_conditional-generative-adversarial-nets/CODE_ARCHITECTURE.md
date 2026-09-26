# CODE_ARCHITECTURE.md — Conditional Generative Adversarial Nets (CGAN)

## Overview

The notebook implements a from-scratch Conditional GAN using PyTorch. Both the generator and discriminator are MLPs that accept concatenated noise+label and image+label respectively. The model is trained on MNIST to generate specific digit classes on demand. After training, a grid of generated digits (one row per class, 0-9) is displayed to demonstrate conditional generation.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, torchvision, matplotlib, numpy. Set random seeds. Configure device (CUDA if available).
- **Key detail**: Only pip-installable packages used (torch, torchvision, matplotlib, numpy).

### 2. Hyperparameters
- **Key values**:
  - `latent_dim = 100` — noise vector dimension (uniform prior)
  - `num_classes = 10` — MNIST digit classes (0-9)
  - `img_dim = 784` — flattened 28×28 image
  - `label_dim = 10` — one-hot label vector
  - `hidden_dim = 256` — hidden layer size for both G and D
  - `lr = 0.0002` — learning rate (Adam)
  - `batch_size = 64` — minibatch size
  - `num_epochs = 100` — training epochs

### 3. Data Loading (MNIST)
- **Purpose**: Load MNIST via torchvision. Images flattened to 784-dim, normalized to [0, 1]. Labels converted to one-hot vectors.
- **Shapes**: 
  - Images: `(batch_size, 1, 28, 28)` → `(batch_size, 784)`
  - Labels: `(batch_size,)` → `(batch_size, 10)` one-hot
- **Custom Dataset**: A wrapper class returns (image_flat, one_hot_label) pairs so the DataLoader yields both simultaneously.

### 4. Generator Network (G)
- **Purpose**: Maps noise + label to a 784-dim image.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 100+10=110)` — concatenated noise z and one-hot label y
  - Hidden: `(batch_size, 256)` — LeakyReLU(0.2)
  - Hidden: `(batch_size, 512)` — LeakyReLU(0.2)
  - Output: `(batch_size, 784)` — Sigmoid (output in [0, 1])
- **Data flow**: `concat(z, y)` → linear → LeakyReLU → linear → LeakyReLU → linear → Sigmoid → fake image (784-dim)
- **Simplification vs paper**: Paper used separate hidden layers for z and y before combining (200→1200 for z, mapped y to separate layer). We concatenate upfront for simplicity — same functional result.

### 5. Discriminator Network (D)
- **Purpose**: Classifies image+label pair as real (1) or fake (0).
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 784+10=794)` — concatenated image and one-hot label
  - Hidden: `(batch_size, 512)` — LeakyReLU(0.2) + Dropout(0.3)
  - Hidden: `(batch_size, 256)` — LeakyReLU(0.2) + Dropout(0.3)
  - Output: `(batch_size, 1)` — Sigmoid (probability)
- **Key insight**: The discriminator sees the label, so it checks not just "is this real?" but "is this a real digit matching label y?"
- **Simplification vs paper**: Paper used maxout activations; we use LeakyReLU for standard practice.

### 6. Loss Functions & Optimizers
- **Loss**: `nn.BCELoss()` — Binary Cross Entropy
- **Optimizers**: Adam for both G and D (lr=0.0002, betas=(0.5, 0.999))
- **Non-saturating generator loss**: Generator maximizes log(D(G(z|y))) using real labels (1.0) for fake samples — stronger gradients than minimizing log(1-D(G(z|y))).

### 7. Training Loop
- **Per-epoch flow**:
  ```
  for each minibatch (x_real, y_real):
    --- Discriminator step ---
    1. Real: D(concat(x_real, y_real)) → BCE(D_real, 1)
    2. Fake: sample z, y_fake; G(concat(z, y_fake)) → x_fake
       D(concat(x_fake, y_fake)) → BCE(D_fake, 0)
    3. D_loss = BCE_real + BCE_fake
    4. Backprop D_loss, step D optimizer
    
    --- Generator step ---
    5. Sample new z, y_fake; G(concat(z, y_fake)) → x_fake
    6. D(concat(x_fake, y_fake)) → BCE(D_fake, 1)  [non-saturating]
    7. G_loss = BCE(D_fake, 1)
    8. Backprop G_loss, step G optimizer
  ```
- **Label sampling for G**: Random labels are sampled for the generator — the generator must learn to produce *any* requested class.
- **Tracking**: D_loss, G_loss logged per epoch. Fixed noise + fixed labels saved for visualizing progression.

### 8. Visualization
- **Class-conditioned grid**: Generate a 10×10 grid — 10 rows (one per digit class 0-9), 10 columns (different noise samples per class). This directly demonstrates conditional generation.
- **Loss curves**: D_loss and G_loss over epochs.
- **Training progression**: Show generated grids at epochs 1, 25, 50, 75, 100 to visualize improvement.

### 9. Conditional Generation Demo
- **Purpose**: After training, demonstrate on-demand generation of specific digits.
- **Method**: Fix noise z, vary the label y from 0-9, show how the same noise produces different digit styles for each class.
- **Also**: Fix label y, vary noise z, show different samples of the same digit class.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `Generator` | nn.Module | (B, 110) | (B, 784) | 3-layer MLP, concat(z,y), LeakyReLU, Sigmoid |
| `Discriminator` | nn.Module | (B, 794) | (B, 1) | 3-layer MLP, concat(x,y), LeakyReLU, Dropout, Sigmoid |
| `BCELoss` | Loss | (B, 1), (B, 1) | scalar | Binary cross-entropy |
| `train()` | Function | — | — | Alternating D/G training loop |
| `generate_grid()` | Function | (10, 100), (10, 10) | matplotlib figure | 10×10 grid of class-conditioned samples |
| `one_hot()` | Function | (B,) | (B, 10) | Convert labels to one-hot vectors |

## Data Flow Diagram

```
Noise z ~ Uniform(-1,1)     Label y (one-hot, dim 10)
        |                           |
        +-----------+---------------+
                    |
              Generator G
                    |
              Fake G(z|y) ──────────────────┐
                                              |
  Real MNIST x ──┐                            |
                 +------ concat(x, y) ───────+
  Label y ───────┘                            |
                                              v
                                    Discriminator D
                                         |        |
                                         v        v
                                    D(real|y)  D(fake|y)
                                         |        |
                                         v        v
                              BCE(D(real|y),1) + BCE(D(fake|y),0) → D_loss
                              BCE(D(fake|y),1)                    → G_loss
                                         |                |
                                         v                v
                                    Update D           Update G
```

## Deliberate Simplifications vs Full Paper

1. **Dataset**: MNIST only — paper also used MIR Flickr 25,000 for multi-modal image tagging
2. **Architecture**: MLPs with LeakyReLU — paper used maxout activations for D, ReLU for G
3. **Concatenation strategy**: We concatenate z+y upfront; paper used separate hidden layers before combining
4. **Optimizer**: Adam (betas=0.5, 0.999) — paper used SGD with momentum and exponential LR decay
5. **No Parzen window evaluation**: Paper's quantitative metric; we use visual inspection
6. **No multi-modal experiment**: Paper's Flickr image-tagging experiment requires pre-trained CNN features + word embeddings; we focus on the MNIST conditional generation demo
7. **Noise prior**: Uniform(-1, 1) — matching the paper's uniform distribution
