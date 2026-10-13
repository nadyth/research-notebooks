# Code Architecture — CLIP (Contrastive Language–Image Pre-training)

## Overview

This notebook implements a simplified CLIP model from scratch in PyTorch, following the core ideas from the paper. It trains a dual image–text encoder with a contrastive loss on a small synthetic dataset of paired simple images and text descriptions, then demonstrates zero-shot classification via text prompts.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Standard PyTorch imports, torchvision for simple image generation
- Device selection (GPU if available, else CPU)
- Reproducibility (random seed)

### 2. Synthetic Dataset Generation
- Creates a small dataset of simple geometric images (colored shapes: circles, squares, triangles in various colors)
- Each image is paired with a text caption like "a red circle" or "a blue square"
- Generates ~2000 training pairs and ~400 test pairs
- Custom `Dataset` class returning (image_tensor, caption_string, label)

### 3. Text Tokenizer (Character-Level)
- Simple character-level tokenizer mapping each character to an integer ID
- Vocabulary built from all unique characters in the caption set
- Pads/truncates captions to a fixed length (e.g., 20 tokens)
- This avoids needing a large pretrained tokenizer (deliberate simplification)

### 4. Image Encoder
- A small CNN: 3 conv blocks (Conv2d → ReLU → MaxPool) → flatten → linear projection
- Input: 3×64×64 images
- Output: 256-dimensional embedding
- Projects to shared embedding space via final linear layer

### 5. Text Encoder
- A small Transformer encoder: token embedding → positional embedding → 2 TransformerEncoderLayers → mean pooling → linear projection
- Input: tokenized caption (batch × seq_len)
- Output: 256-dimensional embedding (same dimension as image encoder)
- Mean pooling over the sequence dimension to get a single vector per caption

### 6. CLIP Model (Combining Both Encoders)
- `CLIPModel` class wraps both encoders
- Computes L2-normalized image and text embeddings
- Returns the cosine similarity matrix (batch_size × batch_size)
- Learnable temperature parameter `logit_scale` (initialized to log(1/0.07) as in paper)

### 7. Contrastive Loss (InfoNCE)
- Given the similarity matrix, the loss is symmetric cross-entropy:
  - For each image, the correct caption should have the highest similarity (row-wise softmax)
  - For each caption, the correct image should have the highest similarity (column-wise softmax)
  - Loss = (loss_image + loss_text) / 2
- Targets are the diagonal indices (0, 1, 2, ..., N-1)

### 8. Training Loop
- Adam optimizer, learning rate 1e-3
- Batch size 64
- Trains for 30 epochs (deliberately small for Kaggle GPU runtime)
- Prints loss every 5 epochs
- Stores loss history for plotting

### 9. Training Visualization
- Plots the contrastive loss curve over epochs
- Shows the loss decreasing, indicating the model is learning to align image–text pairs

### 10. Zero-Shot Classification Demo
- Creates a set of class prompts: "a red circle", "a blue square", "a green triangle", etc.
- Encodes all text prompts to get text embeddings
- For test images, encodes them and computes cosine similarity to all text prompts
- Predicts the class with highest similarity
- Prints accuracy and shows a few example predictions

### 11. Embedding Space Visualization
- Uses t-SNE to project the 256-dim embeddings to 2D
- Plots image embeddings colored by their true class
- Shows that CLIP embeddings cluster by semantic category, not just pixel similarity

## Key Functions/Classes

| Component | Purpose |
|-----------|---------|
| `ShapeDataset` | Generates synthetic image–caption pairs (colored shapes) |
| `char_tokenizer` | Character-level tokenization + padding |
| `ImageEncoder` | Small CNN that maps images → 256-dim embedding |
| `TextEncoder` | Small Transformer that maps text → 256-dim embedding |
| `CLIPModel` | Wraps both encoders, computes similarity matrix with temperature |
| `contrastive_loss` | Symmetric InfoNCE loss (cross-entropy on diagonal) |
| `zero_shot_classify` | Classify images by matching against text prompts |

## Data Flow / Shapes

```
Image (B, 3, 64, 64) → ImageEncoder → (B, 256) → L2-norm → (B, 256)
Caption (B, 20) → TextEncoder → (B, 256) → L2-norm → (B, 256)
Similarity = image_emb @ text_emb.T * temperature → (B, B)
Loss = symmetric_cross_entropy(similarity, diagonal_labels)
```

## Deliberate Simplifications vs Full Paper

1. **Dataset scale:** 2000 pairs vs 400M — synthetic shapes vs real web images
2. **Image encoder:** Small 3-layer CNN vs ResNet-50/ViT-B/32
3. **Text encoder:** 2-layer Transformer with char-level tokenizer vs 12-layer Transformer with BPE
4. **Embedding dimension:** 256 vs 512/768
5. **Batch size:** 64 vs 32768 (large batch is critical for contrastive learning at scale)
6. **Training compute:** Minutes on a single GPU vs hundreds of GPU-days
7. **Prompt engineering:** Basic prompts vs the paper's ensemble of 80 prompt templates
8. **No multi-GPU, no mixed precision, no gradient accumulation** — all used in the original
