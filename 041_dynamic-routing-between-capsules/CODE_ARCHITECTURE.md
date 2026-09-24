# Code Architecture — Dynamic Routing Between Capsules (CapsNet)

## Notebook Structure

The notebook is organized into the following sections, each corresponding to a key component of the Capsule Network architecture as described in the paper.

### Section 1: Setup & Imports
- Install/import `torch`, `torchvision`, `matplotlib`, `numpy`.
- Set random seed for reproducibility.
- Configure device (CPU-friendly: all operations are small-scale and complete in <15 min on CPU).

### Section 2: Data Loading & Preprocessing
- Load MNIST via `torchvision.datasets.MNIST`.
- Apply minimal transforms: `ToTensor()` only (paper uses simple shifts for augmentation; we keep it minimal for CPU).
- Create `DataLoader` with batch_size=128, shuffle=True for train; batch_size=256 for test.
- Visualize a few sample digits.

### Section 3: Squashing Function
- **Function:** `squash(s, dim=-1)`
- Implements Eq. 1: `v_j = (||s_j||² / (1 + ||s_j||²)) * (s_j / ||s_j||)`
- Preserves vector orientation, scales magnitude to [0, 1).
- Key: uses safe normalization to avoid division by zero (epsilon).

### Section 4: PrimaryCaps Layer
- **Class:** `PrimaryCapsule(nn.Module)`
- Wraps a standard `nn.Conv2d` with 256→256 channels, kernel=9, stride=2, producing 8D capsule outputs.
- Input: Conv1 output [B, 256, 20, 20] → Output: [B, 32, 6, 6, 8] (32 types × 6×6 spatial × 8D vector).
- Reshaping: the 256 output channels are grouped into 32 groups of 8 (each group = one 8D capsule).
- No routing between Conv1 and PrimaryCaps (Conv1 output is 1D scalar, nothing to agree on).

### Section 5: DigitCaps Layer with Dynamic Routing
- **Class:** `DigitCapsule(nn.Module)`
- Input: PrimaryCaps output flattened to [B, 1152, 8D] (32 × 6 × 6 = 1152 primary capsules).
- Output: [B, 10, 16D] (10 digit classes, 16D each).
- **Weight matrices:** `W_ij` of shape [10, 1152, 8, 16] — transforms each primary capsule's 8D vector into a 16D prediction for each digit capsule.
- **Routing algorithm (Procedure 1):**
  1. Initialize logits `b_ij = 0` for all i (1152 primary capsules) × j (10 digit capsules).
  2. For r iterations (default 3):
     - `c_ij = softmax(b_ij, dim=j)` — coupling coefficients sum to 1 per primary capsule.
     - `s_j = Σ_i c_ij * û_j|i` — weighted sum of predictions.
     - `v_j = squash(s_j)` — apply squashing non-linearity.
     - `b_ij += û_j|i · v_j` — update logits by agreement (scalar product).
- This is the heart of the paper — iterative, input-dependent routing replaces max-pooling.

### Section 6: Margin Loss
- **Class:** `MarginLoss(nn.Module)`
- For each digit class k: `L_k = T_k * max(0, m+ - ||v_k||)² + λ(1-T_k) * max(0, ||v_k|| - m-)²`
- Parameters: m+ = 0.9, m- = 0.1, λ = 0.5.
- Total loss = sum over all 10 classes.
- Handles multiple overlapping digits (each class independent).

### Section 7: Decoder (Reconstruction Regularizer)
- **Class:** `Decoder(nn.Module)`
- Input: masked DigitCaps output [B, 1, 16D] (only the correct class capsule, zeroed otherwise).
- Architecture: FC(16→512) → ReLU → FC(512→1024) → ReLU → FC(1024→784) → Sigmoid.
- Output reshaped to [B, 1, 28, 28].
- Reconstruction loss: Euclidean distance (MSE) between input image and reconstruction.
- Total loss = margin_loss + 0.0005 * reconstruction_loss (paper uses 0.0005 weight).

### Section 8: Full CapsNet Model
- **Class:** `CapsNet(nn.Module)`
- Combines: Conv1 → PrimaryCaps → DigitCaps → Decoder.
- Conv1: `nn.Conv2d(1, 256, kernel_size=9, stride=1)` + ReLU.
- Forward pass returns: (digit_caps_output, reconstruction).
- Masking during training: keep only the correct (or predicted) class capsule output for reconstruction.

### Section 9: Training Loop
- Optimizer: Adam, lr=1e-3, weight_decay=0 (paper uses Adam with default params + exponentially decaying lr; we use fixed lr for simplicity).
- Epochs: 10 (reduced from paper's larger setup for CPU friendliness; paper achieves 0.25% with 10K steps).
- For each batch: forward → margin loss + reconstruction loss → backward → step.
- Print training loss and accuracy every N batches.
- Track test accuracy after each epoch.

### Section 10: Evaluation & Visualization
- Final test accuracy report.
- **Pose visualization:** Perturb each of the 16 dimensions of a DigitCaps output by intervals of 0.05 in [-0.25, 0.25], feed through decoder, and display the reconstruction grid. This shows what each dimension encodes (thickness, scale, skew, etc.) — replicating Figure 4 in the paper.
- **Reconstruction grid:** Show side-by-side input, prediction, and reconstruction for sample test images — replicating Figure 3.

## Data Flow & Shapes

```
Input Image: [B, 1, 28, 28]
    ↓ Conv1 (256×9×9, stride=1, ReLU)
Conv1 Output: [B, 256, 20, 20]
    ↓ PrimaryCaps (Conv2d 256→256, k=9, s=2, reshape to 8D capsules)
PrimaryCaps: [B, 32, 6, 6, 8] → reshape [B, 1152, 8]
    ↓ W_ij transform [B, 1152, 8] → [B, 10, 1152, 16]
    ↓ Dynamic Routing (3 iterations)
DigitCaps: [B, 10, 16]
    ↓ Mask (keep correct class)
Masked: [B, 1, 16]
    ↓ Decoder (FC 16→512→1024→784)
Reconstruction: [B, 1, 28, 28]
    ↓ Loss = MarginLoss(||v_k||) + 0.0005 * MSE(recon, input)
```

## Deliberate Simplifications vs Full Paper

1. **Epochs/Steps:** 10 epochs (~4.7K steps at batch_size=128) vs the paper's larger training. Sufficient to reach ~99% accuracy on MNIST.
2. **Learning rate:** Fixed lr=1e-3 instead of the paper's exponentially decaying lr. Simpler, still converges.
3. **Data augmentation:** Only standard MNIST normalization. Paper uses ±2 pixel shifts; omitted for CPU speed.
4. **No MultiMNIST/overlapping digits experiment:** The overlapping-digit segmentation experiment requires the MultiMNIST dataset generation, which is omitted to keep the notebook focused and CPU-friendly. The core capsule + routing mechanism is fully implemented.
5. **affNIST robustness test:** Omitted (requires the affNIST dataset).
6. **Batch size:** 128 (paper doesn't specify; common choice in implementations).
