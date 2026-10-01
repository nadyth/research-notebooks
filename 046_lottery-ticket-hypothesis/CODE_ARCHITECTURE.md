# Code Architecture — Lottery Ticket Hypothesis Notebook

## Overview

The notebook implements **Iterative Magnitude Pruning (IMP)** from scratch to extract "winning ticket" subnetworks from a trained MLP on MNIST. It then compares the winning ticket (original initialization + pruning mask) against a re-initialized subnetwork (new random weights + same mask) to demonstrate the paper's core finding.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, torchvision (MNIST), matplotlib, numpy
- Reproducibility: fixed random seeds for all RNGs

### 2. Data Loading
- MNIST dataset via torchvision
- Simple DataLoader with batch_size=128
- Train/test split as provided by torchvision

### 3. Model Definition — Simple MLP
- Architecture: `784 → 256 → 128 → 10` (fully-connected)
- ReLU activations between hidden layers
- This is deliberately small to keep IMP iterations fast on GPU within the 30-min Kaggle limit

### 4. Training Function `train_model(model, mask, train_loader, epochs, lr)`
- Accepts an optional binary `mask` (dict of per-layer boolean tensors) that zeros out pruned weights' gradients during training
- Cross-entropy loss, Adam optimizer (lr=1e-3)
- Returns trained model and training history (loss, test accuracy per epoch)

### 5. Pruning Function `prune_by_magnitude(model, prune_fraction)`
- For each linear layer, compute the absolute weight values
- Find the threshold at the `prune_fraction` percentile
- Create a binary mask: `True` for weights above threshold, `False` for pruned
- Returns the mask dict and the pruning percentage applied so far

### 6. Mask Application `apply_mask(model, mask)`
- Zeroes out pruned weights in the model parameters
- Called after each pruning iteration to maintain sparsity

### 7. Iterative Magnitude Pruning Loop
```
for iteration in range(num_prune_iterations):
    1. Save initial weights θ₀ (iteration 0 only)
    2. Train the model with current mask
    3. Prune prune_fraction (20%) of remaining weights → new mask
    4. Reset surviving weights to θ₀ (original initialization)
    5. Log: sparsity %, final accuracy, training curve
```
- `num_prune_iterations = 5` (prunes 20% each → ~33% remaining after 5 rounds)
- Each iteration trains for `epochs_per_iteration = 5`

### 8. Winning Ticket Evaluation
- Take the final mask (e.g., ~33% of weights remaining)
- **Winning Ticket:** Reset surviving weights to θ₀, retrain from scratch
- **Random Re-init Control:** Same mask, but re-initialize surviving weights with *new* random values, retrain
- Compare training curves (loss, accuracy) side by side

### 9. Visualization
- **Plot 1:** Test accuracy vs. epochs for full model, winning ticket, and random-reinit
- **Plot 2:** Sparsity vs. accuracy curve across pruning iterations
- **Plot 3:** Training loss comparison

## Key Functions/Classes

| Function | Purpose |
|---|---|
| `SimpleMLP` | 3-layer MLP (784→256→128→10) |
| `train_model(model, mask, ...)` | Trains with optional pruning mask |
| `prune_by_magnitude(model, fraction)` | Global unstructured magnitude pruning |
| `apply_mask(model, mask)` | Zeroes pruned weights |
| `reset_to_initial(model, init_state)` | Restores original weights for surviving connections |
| `iterative_magnitude_pruning(...)` | Full IMP loop |

## Data Flow / Shapes

- Input: `(batch, 1, 28, 28)` → flattened to `(batch, 784)`
- FC1: `(batch, 784) @ (784, 256)` → `(batch, 256)`
- FC2: `(batch, 256) @ (256, 128)` → `(batch, 128)`
- FC3: `(batch, 128) @ (128, 10)` → `(batch, 10)` → softmax

## Deliberate Simplifications vs. Full Paper

1. **MLP only, no CNN/ResNet:** The paper uses Lenet-style CNNs and ResNets; we use a simple MLP for speed.
2. **5 pruning iterations (20% each):** Paper uses more iterations with finer granularity. We keep iterations low to fit the Kaggle 30-min limit.
3. **No weight rewinding (late resetting):** The paper uses weight rewinding for deep networks (reset to iteration k, not 0). Our MLP is shallow enough that reset-to-zero works.
4. **Adam optimizer:** Paper uses SGD with momentum and learning rate schedules; we use Adam for faster convergence in fewer epochs.
5. **MNIST only:** Paper also experiments with CIFAR-10 and ImageNet.
6. **Global unstructured pruning:** We prune individual weights (unstructured), same as the paper's primary experiments, rather than structured (channel/filters) pruning.
