# Code Architecture — U-Net Notebook

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import PyTorch, torchvision, numpy, matplotlib
- Set device (CUDA if available)
- Install any needed pip packages

### 2. Synthetic Dataset Generation
Since the original paper uses proprietary biomedical EM data, we generate a **synthetic cell-like segmentation dataset**:
- Create images with overlapping circular blobs (simulating cells) on noisy backgrounds
- Generate corresponding binary masks (1 where cell, 0 background)
- Include touching/overlapping cells to test boundary separation
- Dataset size: ~200 training images, 50 validation
- Image size: 128×128 (scaled down from 512×512 for Kaggle GPU feasibility)

### 3. Data Augmentation
- Random horizontal/vertical flips
- Random rotations (90°, 180°, 270°)
- Elastic deformations (simplified — using affine + grid deformation)
- Custom PyTorch Dataset class with on-the-fly augmentation

### 4. U-Net Architecture (from scratch)
```
Encoder (Contracting Path):
  Block 1: Conv(1→64) → Conv(64→64) → ReLU → MaxPool   [128×128 → 64×64]
  Block 2: Conv(64→128) → Conv(128→128) → ReLU → MaxPool [64×64 → 32×32]
  Block 3: Conv(128→256) → Conv(256→256) → ReLU → MaxPool [32×32 → 16×16]
  Block 4: Conv(256→512) → Conv(512→512) → ReLU → MaxPool [16×16 → 8×8]

Bottleneck:
  Conv(512→1024) → Conv(1024→1024) → ReLU                [8×8]

Decoder (Expanding Path):
  Up1: UpConv(1024→512) + skip(Block4) → Conv(1024→512) → Conv(512→512)  [8→16]
  Up2: UpConv(512→256) + skip(Block3) → Conv(512→256) → Conv(256→256)    [16→32]
  Up3: UpConv(256→128) + skip(Block2) → Conv(256→128) → Conv(128→128)    [32→64]
  Up4: UpConv(128→64) + skip(Block1) → Conv(128→64) → Conv(64→64)        [64→128]

Output: Conv(64→2) — 1×1 conv producing 2-class segmentation map
```

Key classes:
- `DoubleConv(nn.Module)`: Two conv-batchnorm-ReLU blocks (batchnorm added vs original, which used only ReLU — a deliberate modernization)
- `DownBlock(nn.Module)`: MaxPool + DoubleConv
- `UpBlock(nn.Module)`: Upsample/UpConv + concatenate skip + DoubleConv
- `UNet(nn.Module)`: Full encoder-decoder assembly

### 5. Loss Function
- **Weighted cross-entropy** with border-aware weights (simplified)
- Alternative: Dice loss (common in segmentation, not in original paper but widely used with U-Net)
- We implement both and compare

### 6. Training Loop
- Optimizer: Adam, learning rate 1e-4
- Epochs: 15 (reduced from original for Kaggle time limit)
- Batch size: 8
- Track training loss, validation loss, and IoU (Intersection over Union) per epoch
- Learning rate scheduler: ReduceLROnPlateau

### 7. Evaluation & Visualization
- Plot training vs validation loss curves
- Plot IoU improvement over epochs
- Show input image / predicted mask / ground truth triplets for multiple samples
- Visualize intermediate feature maps from encoder and decoder layers

## Data Flow / Tensor Shapes
```
Input:  [B, 1, 128, 128]   (grayscale synthetic cell images)
  → Encoder Block 1: [B, 64, 64, 64]    (skip saved)
  → Encoder Block 2: [B, 128, 32, 32]   (skip saved)
  → Encoder Block 3: [B, 256, 16, 16]   (skip saved)
  → Encoder Block 4: [B, 512, 8, 8]     (skip saved)
  → Bottleneck:      [B, 1024, 8, 8]
  → Decoder Up 1:    [B, 512, 16, 16]   (concat with skip4)
  → Decoder Up 2:    [B, 256, 32, 32]   (concat with skip3)
  → Decoder Up 3:    [B, 128, 64, 64]   (concat with skip2)
  → Decoder Up 4:    [B, 64, 128, 128]  (concat with skip1)
  → Output conv:     [B, 2, 128, 128]   (per-pixel class scores)
```

## Deliberate Simplifications vs Full Paper

1. **Image size:** 128×128 instead of 512×512 — faster training on Kaggle GPU
2. **Synthetic data:** Generated blob shapes instead of real EM/light microscopy data (proprietary datasets)
2. **BatchNorm added:** Original used plain conv+ReLU; we add BatchNorm for training stability (standard modern practice)
3. **Epochs:** 15 instead of full training (original trained until convergence over many hours)
4. **Border weight map simplified:** We use a simplified version of the border-aware weight computation
5. **Dice loss added:** Not in original paper but now standard with U-Net — included for comparison
6. **Overlap-tile strategy omitted:** Not needed with 128×128 images that fit in memory
7. **No test-time augmentation:** Original paper used sliding window with overlap and test-time augmentation; omitted for simplicity
