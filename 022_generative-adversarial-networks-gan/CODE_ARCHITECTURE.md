# CODE_ARCHITECTURE.md — Generative Adversarial Networks (GAN)

## Overview

The notebook implements a from-scratch GAN using the original paper's framework: a generator MLP and a discriminator MLP trained in an adversarial minimax loop on MNIST. The goal is to generate handwritten digits that progressively improve over training epochs, with a visual animation of generated samples.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, torchvision (for MNIST), matplotlib, numpy. Set random seeds for reproducibility. Configure device (CUDA if available).
- **Key detail**: Only pip-installable packages used (torch, torchvision, matplotlib, numpy). No external data downloads beyond torchvision's built-in MNIST loader.

### 2. Hyperparameters
- **Purpose**: Centralize all training configuration.
- **Key values**:
  - `latent_dim = 100` — dimension of noise vector z (uniform prior)
  - `img_dim = 784` — flattened MNIST image (28×28)
  - `hidden_dim = 256` — hidden layer size for both G and D
  - `lr = 0.0002` — learning rate (following common GAN practice)
  - `batch_size = 64` — minibatch size
  - `num_epochs = 200` — training epochs (enough to see clear improvement)
  - `k_steps = 1` — discriminator updates per generator update (as in the paper)
- **Data flow**: These constants flow into model construction and the training loop.

### 3. Data Loading (MNIST)
- **Purpose**: Load MNIST dataset via torchvision, flatten images to 784-dim vectors, normalize to [0, 1].
- **Key functions**: `torchvision.datasets.MNIST`, `DataLoader`
- **Shapes**: 
  - Raw MNIST: `(batch_size, 1, 28, 28)` → reshaped to `(batch_size, 784)`
  - Normalized: pixel values in [0, 1] (divided by 255)
- **Simplification vs paper**: The paper also tested on TFD, CIFAR-10, and SVHN. We use only MNIST for clarity and speed.

### 4. Generator Network (G)
- **Purpose**: Maps noise vector z to a 784-dim image vector.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 100)` — latent vector
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2)
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2)
  - Output: `(batch_size, 784)` — with Sigmoid (output in [0, 1])
- **Key class**: `Generator(nn.Module)` with `forward(z)` method
- **Data flow**: `z ~ Uniform(-1, 1)` → linear → LeakyReLU → linear → LeakyReLU → linear → Sigmoid → fake image (784-dim)
- **Simplification vs paper**: The paper used a mix of ReLU (generator) and maxout (discriminator). We use LeakyReLU throughout for training stability, which is now standard practice.

### 5. Discriminator Network (D)
- **Purpose**: Classifies input image as real (1) or fake (0).
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 784)` — image vector
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2) + Dropout(0.3)
  - Hidden: `(batch_size, 256)` — with LeakyReLU(0.2) + Dropout(0.3)
  - Output: `(batch_size, 1)` — with Sigmoid (probability)
- **Key class**: `Discriminator(nn.Module)` with `forward(x)` method
- **Data flow**: image → linear → LeakyReLU → Dropout → linear → LeakyReLU → Dropout → linear → Sigmoid → probability scalar
- **Simplification vs paper**: Paper used maxout activations in D; we use LeakyReLU. Dropout included as the paper mentions dropout for regularization.

### 6. Loss Functions & Optimizers
- **Purpose**: Define the adversarial loss (BCE) and optimizers for both networks.
- **Key details**:
  - **Loss**: `nn.BCELoss()` — Binary Cross Entropy, since D outputs a probability and the objective is log D(x) + log(1 - D(G(z)))
  - **Real label**: 1.0, **Fake label**: 0.0
  - **Optimizer**: Adam for both G and D (lr=0.0002, betas=(0.5, 0.999)) — the beta1=0.5 setting is standard for GANs to reduce momentum oscillation
- **Non-saturating generator loss**: Instead of minimizing log(1 - D(G(z))), we maximize log(D(G(z))) by using real labels (1.0) for fake samples — this is the practical trick from the paper that provides stronger gradients early in training.

### 7. Training Loop
- **Purpose**: Alternating D and G updates following Algorithm 1 from the paper.
- **Per-epoch flow**:
  ```
  for each minibatch:
    --- Discriminator step (k=1) ---
    1. Sample real batch x from MNIST
    2. Sample noise z, generate fake batch G(z)
    3. D_loss = -[mean(log D(x)) + mean(log(1 - D(G(z))))]
       (implemented as BCE on real + BCE on fake)
    4. Backprop D_loss, step D optimizer
    
    --- Generator step ---
    5. Sample new noise z, generate fake batch G(z)
    6. G_loss = -mean(log D(G(z)))  [non-saturating]
       (implemented as BCE with real labels on fake data)
    7. Backprop G_loss, step G optimizer
  ```
- **Tracking**: D_loss, G_loss logged per epoch. Fixed noise vector saved for visualizing progression.
- **Key detail**: A fixed noise vector `fixed_z` is used at each epoch to generate and save samples, so we can animate the same random digits improving over time.

### 8. Visualization & Animation
- **Purpose**: Show generated digits at regular intervals and create an animation of improvement.
- **Key functions**:
  - `show_generated(generator, z)` — grid of generated digits using matplotlib
  - `animate_training(saved_images)` — creates a GIF/animation from saved epoch snapshots
- **Data flow**: Fixed noise → Generator → reshape to (28, 28) → matplotlib grid
- **Output**: 
  - Loss curves (D_loss and G_loss over epochs)
  - Grid of generated digits at epochs 1, 50, 100, 150, 200
  - Animation showing the progression

### 9. Evaluation (Optional)
- **Purpose**: Qualitative assessment of generated samples.
- **Method**: Visual inspection of generated digit grids — no quantitative metric (like Inception Score or FID) in the original paper.
- **Simplification**: The original paper used a Parzen window-based log-likelihood estimate for quantitative evaluation. We skip this as it is known to be an unreliable metric and is not the key contribution.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `Generator` | nn.Module | (B, 100) | (B, 784) | 3-layer MLP, LeakyReLU, Sigmoid output |
| `Discriminator` | nn.Module | (B, 784) | (B, 1) | 3-layer MLP, LeakyReLU, Dropout, Sigmoid |
| `BCELoss` | Loss | (B, 1), (B, 1) | scalar | Binary cross-entropy for adversarial training |
| `train()` | Function | — | — | Alternating D/G training loop with loss tracking |
| `show_generated()` | Function | (B, 100) | matplotlib figure | Visualizes generator output grid |
| `animate_training()` | Function | list of (B, 784) | animation | GIF showing digit improvement over epochs |

## Data Flow Diagram

```
Noise z ~ Uniform(-1,1)                      Real MNIST images x
        |                                              |
        v                                              |
   Generator G                                         |
        |                                              |
        v                                              v
  Fake G(z) ──────────────────────────────────► Discriminator D
                                                   |        |
                                                   v        v
                                              D(real)    D(fake)
                                                   |        |
                                                   v        v
                                              BCE(D(real), 1) + BCE(D(fake), 0)  → D_loss
                                              BCE(D(fake), 1)                     → G_loss (non-saturating)
                                                   |                |
                                                   v                v
                                              Update D           Update G
```

## Deliberate Simplifications vs Full Paper

1. **Dataset**: MNIST only — paper also used TFD, CIFAR-10, SVHN
2. **Architecture**: MLPs with LeakyReLU — paper used maxout activations for D, ReLU for G
3. **No Parzen window evaluation**: Paper's quantitative metric is known to be unreliable; we rely on visual inspection
4. **Adam optimizer**: Paper used SGD with momentum; Adam with betas=(0.5, 0.999) is now the GAN standard
5. **Noise prior**: Uniform(-1, 1) — paper used uniform prior; many later works use Gaussian
6. **No dropout in G**: Paper applied dropout in both; we apply only in D for stability
7. **k=1**: Matches the paper's choice (one D step per G step)
