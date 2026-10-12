# Code Architecture: Vision Transformer (ViT) from Scratch

## Overview

The notebook implements a miniature Vision Transformer from scratch using PyTorch, following the architecture described in "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020). The model is trained on a subset of CIFAR-10 to demonstrate the core mechanics of patch embedding, positional encoding, multi-head self-attention, and the [CLS] token classification paradigm.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `torch`, `torchvision`, `matplotlib`, `numpy`
- Sets random seeds for reproducibility
- Detects CUDA device (GPU) or falls back to CPU

### 2. Data Loading & Preprocessing
- Uses CIFAR-10 (32×32 RGB images, 10 classes) from torchvision
- Applies standard normalization with horizontal flip augmentation
- Uses a 20K-image subset of the training set for faster training
- Creates train/test DataLoaders with batch_size=256
- **Simplification:** Uses a CIFAR-10 subset instead of ImageNet-21k or JFT-300M. The original paper pre-trains on massive datasets; we train from scratch on a small subset to demonstrate the architecture within Kaggle's GPU/time constraints.

### 3. Patch Embedding Layer
- **Class `PatchEmbedding`:** Takes input images of shape (B, C, H, W), splits into non-overlapping P×P patches, flattens, and linearly projects to D dimensions
- Uses `nn.Conv2d` with kernel_size=stride=P as an efficient patch extraction + projection
- Output shape: (B, N, D) where N = (H/P) × (W/P)
- **Key shapes:** Input (B, 3, 32, 32) → with P=4 → N=64 patches → Output (B, 64, 128)

### 4. Multi-Head Self-Attention (MSA)
- **Class `MultiHeadSelfAttention`:** Standard scaled dot-product attention with multiple heads
- Input: (B, N+1, D) — includes the [CLS] token
- Projects to Q, K, V via a single QKV linear layer, splits into `num_heads` heads
- Attention: softmax(QK^T / sqrt(d_k)) V
- Concatenates heads and projects back to D
- Stores attention weights for visualization

### 5. Transformer Encoder Block
- **Class `TransformerEncoderBlock`:** LayerNorm → MSA → residual; LayerNorm → MLP(GELU) → residual
- MLP: two linear layers with GELU activation, hidden dim = 2×D (reduced from 4×D for faster training)
- Pre-LayerNorm architecture (as in the paper)

### 6. Vision Transformer
- **Class `VisionTransformer`:** Assembles all components
  - `PatchEmbedding`: image → patch tokens
  - `cls_token`: learnable (1, 1, D) parameter prepended to sequence
  - `pos_embed`: learnable (1, N+1, D) positional embedding
  - `nn.Sequential` of `TransformerEncoderBlock` × L layers
  - `LayerNorm` after encoder
  - `Linear(D, num_classes)` classification head on [CLS] token output
- `get_attention_maps()`: extracts CLS-to-patch attention from each layer for visualization

### 7. Training Loop
- Cross-entropy loss, AdamW optimizer, cosine annealing LR scheduler
- Trains for 10 epochs on 20K images
- Tracks and plots training/test loss and accuracy curves

### 8. Patch Attention Visualization
- Extracts attention weights from the [CLS] token to all patch tokens
- Reshapes attention weights to spatial grid (8, 8)
- Upsamples to original image size and overlays as heatmap
- Visualizes which image patches the model attends to across layers

### 9. Positional Embedding Similarity
- Computes cosine similarity between learned positional embeddings
- Shows that nearby patches develop similar embeddings (spatial structure emerges)

## Key Functions/Classes

| Component | Purpose | Input → Output |
|-----------|---------|----------------|
| `PatchEmbedding` | Split image into patches, project to D dims | (B,C,H,W) → (B,N,D) |
| `MultiHeadSelfAttention` | Scaled dot-product attention with H heads | (B,N+1,D) → (B,N+1,D) |
| `TransformerEncoderBlock` | One encoder layer with MSA + MLP | (B,N+1,D) → (B,N+1,D) |
| `VisionTransformer` | Full ViT model | (B,C,H,W) → (B,num_classes) |

## Deliberate Simplifications vs Full Paper

1. **Dataset:** CIFAR-10 20K subset vs ImageNet-21k/JFT-300M (14M–300M images). No pre-training; trained from scratch.
2. **Model size:** ViT-Tiny (L=4, D=128, heads=4, ~0.5M params) vs ViT-Base (L=12, D=768, 86M params). Scaled down for GPU memory and time constraints.
3. **Patch size:** P=4 (for 32×32 images) vs P=16 (for 224×224 images in the paper). Same principle, adjusted for smaller resolution.
4. **MLP ratio:** 2×D vs 4×D in the paper, for faster training.
5. **No pre-training/fine-tuning:** The paper's key contribution is pre-training on massive datasets then fine-tuning. We train end-to-end on CIFAR-10 subset only.
6. **Epochs:** 10 vs hundreds in the paper. Demonstrates architecture, not SOTA results.
7. **1D positional embeddings:** Same as the paper; no 2D-aware variants explored.

## Data Flow

```
Image (B, 3, 32, 32)
  → PatchEmbedding: (B, 64, 128)         [P=4, N=64, D=128]
  → prepend CLS token: (B, 65, 128)
  → add positional embedding: (B, 65, 128)
  → Transformer Encoder × 4: (B, 65, 128)
  → LayerNorm: (B, 65, 128)
  → extract CLS: (B, 128)
  → Linear classifier: (B, 10)
  → softmax → prediction
```
