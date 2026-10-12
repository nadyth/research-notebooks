# Code Architecture — DALL-E (Zero-Shot Text-to-Image Generation)

## Overview

This notebook implements a simplified DALL-E pipeline from scratch in PyTorch, following the core two-stage architecture from the paper: (1) a discrete VAE (dVAE) that tokenizes images into a grid of discrete codebook entries, and (2) an autoregressive Transformer that models the joint sequence of text tokens + image tokens. We train on a small synthetic dataset of colored shapes with text captions, then sample new images from text prompts. The implementation is deliberately scaled down to run within Kaggle's GPU time limits.

## Section-by-Section Breakdown

### 1. Setup & Imports
- PyTorch, torchvision, numpy, matplotlib
- Device selection (GPU if available)
- Reproducibility (random seed)
- Hyperparameter configuration in a single cell

### 2. Synthetic Dataset Generation
- Creates simple 64×64 RGB images containing colored geometric shapes (circles, squares, triangles) in 6 colors
- Each image is paired with a text caption like "a red circle" or "a blue square on a white background"
- Generates ~3000 training pairs and ~600 test pairs
- Custom `Dataset` class returning (image_tensor, caption_string)
- This is a deliberate simplification — the real DALL-E uses 250M internet image-text pairs at 256×256

### 3. Character-Level Tokenizer
- Maps each character to an integer ID (vocabulary ~30 chars)
- Pads/truncates captions to a fixed length (e.g., 30 tokens)
- Real DALL-E uses BPE with 16384 vocabulary; we use character-level for simplicity

### 4. Discrete VAE (dVAE)
- **Encoder:** 4 conv layers (Conv2d → ReLU → BatchNorm → downsample) mapping 64×64×3 → 16×16×latent_dim
- **Codebook:** 256 entries of dimension 64 (real DALL-E uses 8192 entries of dimension 8192)
- **Vector Quantization:** nearest-neighbor lookup in codebook; straight-through gradient estimator (gradient passes through the quantization step unchanged)
- **Decoder:** 4 transposed conv layers mapping 16×16×64 → 64×64×3
- **Loss:** reconstruction (MSE) + commitment loss (β * MSE(z_e, sg(z_q))) + codebook loss (MSE(sg(z_e), z_q))
- Trained independently for 15 epochs before the Transformer stage
- Output: 16×16 = 256 image tokens per image (real DALL-E: 32×32 = 1024 tokens)

### 5. dVAE Training & Visualization
- Trains the dVAE on the synthetic image dataset
- Visualizes original vs reconstructed images to verify the codebook is learning useful representations
- After training, encodes the entire training set to get image token sequences for Transformer training

### 6. Autoregressive Transformer
- **Architecture:** decoder-only Transformer (6 layers, 256 dim, 8 heads, 1024 FFN dim)
- **Vocabulary:** text_vocab_size + image_codebook_size + special tokens (BOS, SEP, EOS, PAD)
- **Input format:** [BOS] + text_tokens + [SEP] + image_tokens + [EOS]
- **Causal attention mask:** standard GPT-style — each token attends only to preceding tokens
- **Positional encoding:** learned positional embeddings
- Real DALL-E uses 12B parameters, 64 layers, 1792 dim; our version is ~2M parameters

### 7. Transformer Training
- Pre-encodes all training images through the frozen dVAE to get image token sequences
- Each training sample: concatenate text tokens + image tokens, apply causal mask, predict next token
- Loss: standard cross-entropy on the full sequence (text + image tokens)
- Adam optimizer, lr 3e-4, batch size 64, 20 epochs
- Text tokens are included in the loss (the model learns to model text too, though we only use the image generation part at inference)

### 8. Training Visualization
- Plots the Transformer loss curve over epochs
- Shows the dVAE reconstruction quality
- Both should show converging behavior

### 9. Text-to-Image Generation (Sampling)
- Given a text prompt: tokenize → [BOS] + text_tokens + [SEP]
- Autoregressively sample image tokens one by one using top-k sampling (k=10) with temperature
- Decode the 256 image tokens through the dVAE decoder to get a 64×64 image
- Generate multiple samples and display them in a grid

### 10. Zero-Shot Generation Demo
- Tests the model on prompts that combine concepts in novel ways (e.g., "a green triangle" if only circles and squares were in training data)
- Displays generated images alongside their text prompts
- Shows that the model can compose visual concepts from text descriptions

## Key Functions/Classes

| Component | Purpose |
|-----------|---------|
| `ShapeDataset` | Generates synthetic image–caption pairs (colored shapes) |
| `CharTokenizer` | Character-level tokenization + padding |
| `dVAE` | Discrete VAE: encoder + codebook + decoder with VQ bottleneck |
| `VectorQuantizer` | Nearest-neighbor codebook lookup with straight-through gradient |
| `DalleTransformer` | Decoder-only Transformer over joint text+image vocabulary |
| `generate_image` | Autoregressive sampling: text → image tokens → dVAE decode → image |
| `train_dvae` | Training loop for the discrete VAE stage |
| `train_transformer` | Training loop for the autoregressive Transformer stage |

## Data Flow / Shapes

```
Image (3, 64, 64)
  → dVAE Encoder → (64, 16, 16)    # continuous latent
  → Vector Quantize → (16, 16)      # discrete token indices (256 tokens)
  → dVAE Decoder → (3, 64, 64)      # reconstructed image

Text caption "a red circle"
  → CharTokenizer → (30,)           # token IDs, padded
  → [BOS] + text_tokens + [SEP] + image_tokens(256) + [EOS] → (288,)
  → Transformer (causal) → logits (288, vocab_size)
  → Cross-entropy loss on next-token prediction

Generation:
  "a blue square" → tokenize → [BOS] + text + [SEP]
  → autoregressive sample 256 image tokens
  → dVAE Decoder → (3, 64, 64) → display
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper (DALL-E) | This Notebook |
|--------|----------------|---------------|
| Image size | 256×256 | 64×64 |
| dVAE codebook | 8192 entries | 256 entries |
| Image tokens | 32×32 = 1024 | 16×16 = 256 |
| Text tokenizer | BPE, vocab 16384 | Char-level, vocab ~30 |
| Transformer params | 12 billion | ~2 million |
| Transformer layers | 64 | 6 |
| d_model | 1792 | 256 |
| Training data | 250M image-text pairs | ~3000 synthetic pairs |
| CLIP reranking | Yes (separate model) | No (display all samples) |
| Training time | Days on GPU clusters | ~15 min on Kaggle GPU |
| Quality | Photorealistic | Simple colored shapes |
