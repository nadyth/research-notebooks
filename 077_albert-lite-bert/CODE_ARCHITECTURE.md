# Code Architecture — 077 ALBERT

## Overview

This notebook implements a from-scratch BERT-style Transformer encoder and introduces the two key ALBERT parameter-reduction techniques:

1. **Standard BERT** — full embedding matrix (V×H), separate parameters per layer, NSP loss
2. **ALBERT** — factorized embedding (V×E + E×H), shared parameters across all layers, SOP loss

Both models are pretrained with Masked Language Modeling (MLM) on the same toy corpus. The notebook compares total parameter counts and pretraining loss curves to demonstrate that ALBERT achieves comparable or better loss with dramatically fewer parameters — the core finding of the paper.

Models are intentionally small (4 Transformer layers, 128 hidden dim, 4 attention heads, ~500 vocabulary, embedding dim 64) to run within Kaggle's 30-minute GPU limit while demonstrating all core ALBERT mechanisms.

## Section-by-Section Breakdown

### 1. Imports & Configuration
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `math`, `random`, `collections`, `matplotlib.pyplot`, `numpy`
- Hyperparameters: `d_model=128`, `n_heads=4`, `n_layers=4`, `vocab_size` (built from corpus), `max_len=64`, `d_embedding=64` (ALBERT factorized embedding dim), `dropout=0.1`
- BERT config: separate parameters per layer, full V×H embedding, NSP loss
- ALBERT config: shared parameters across layers, factorized V×E + E×H embedding, SOP loss
- Both: `lr=5e-4`, `epochs=20`, `batch_size=32`

### 2. Toy Corpus Construction
- A small collection of English sentences organised into short paragraph documents (2-4 sentences each)
- For NSP training (BERT): sentence pairs (A, B) where B is either the actual next sentence or a random sentence from the corpus
- For SOP training (ALBERT): sentence pairs (A, B) where B is either the actual next sentence or the *same pair swapped* (B, A)
- Word-level tokenizer (split on whitespace, lowercase) — deliberately simpler than WordPiece/BPE for clarity

### 3. Vocabulary Building
- Build vocabulary from all words in the toy corpus
- Special tokens: `[PAD]=0`, `[UNK]=1`, `[CLS]=2`, `[SEP]=3`, `[MASK]=4`
- Word-to-id and id-to-word mappings (~500 tokens)

### 4. Masking Function (MLM)
- Shared by both BERT and ALBERT — identical MLM masking
- Select 15% of tokens: 80% → `[MASK]`, 10% → random token, 10% → unchanged
- MLM labels: original token at masked positions, -100 elsewhere (PyTorch convention)
- Returns: masked_input_ids, mlm_labels

### 5. Dataset Construction

#### 5a. BERT Dataset (with NSP)
- Each example: `[CLS] A [SEP] B [SEP]` with segment IDs (0 for A, 1 for B)
- NSP label: 1 if B follows A in the corpus, 0 if random
- MLM labels computed via masking function
- Returns: (input_ids, segment_ids, attention_mask, mlm_labels, nsp_label)

#### 5b. ALBERT Dataset (with SOP)
- Each example: `[CLS] A [SEP] B [SEP]` with segment IDs (0 for A, 1 for B)
- SOP label: 1 if (A, B) are in correct order, 0 if swapped
- MLM labels computed via masking function
- Returns: (input_ids, segment_ids, attention_mask, mlm_labels, sop_label)

### 6. Model Architecture

#### 6a. Multi-Head Self-Attention
- Standard scaled dot-product attention: Q, K, V projections per head
- `attention = softmax(QK^T / sqrt(d_k)) V`
- Multi-head: `n_heads` parallel attention heads, outputs concatenated and projected
- This is identical for both BERT and ALBERT — the architecture is the same; only parameter usage differs

#### 6b. Transformer Encoder Layer
- Each layer: Multi-Head Attention → Add & LayerNorm → Feed-Forward (2 linear layers with GELU) → Add & LayerNorm
- Feed-forward: `Linear(d_model, d_ff)` → GELU → `Linear(d_ff, d_model)` where `d_ff = 4 * d_model`
- **BERT**: each layer has its own instance of this module (separate weights)
- **ALBERT**: a single shared instance is applied N times (same weights reused)

#### 6c. BERT Model (standard)
- **Embedding**: `nn.Embedding(vocab_size, d_model)` — full V×H matrix + positional embedding + segment embedding
- **Encoder**: `nn.ModuleList([TransformerLayer() for _ in range(n_layers)])` — each layer has unique weights
- **NSP head**: `Linear(d_model, 2)` — binary classification (is_next / not_next)
- **MLM head**: `Linear(d_model, vocab_size)` — predict original tokens

#### 6d. ALBERT Model (parameter-reduced)
- **Factorized Embedding**: `nn.Embedding(vocab_size, d_embedding)` → `Linear(d_embedding, d_model)` — V×E + E×H instead of V×H
- **Encoder**: a single `TransformerLayer()` applied `n_layers` times in a loop — all layers share the same weights
- **SOP head**: `Linear(d_model, 2)` — binary classification (correct_order / swapped)
- **MLM head**: `Linear(d_model, vocab_size)` — same as BERT

### 7. Parameter Count Comparison
- Count and print total parameters for BERT vs ALBERT
- Break down by component: embedding, encoder, heads
- Expected: ALBERT embedding ~4× smaller (factorized), ALBERT encoder ~4× smaller (shared across 4 layers)
- Print a comparison table

### 8. Training Loop
- Standard training loop for both models:
  - Forward pass: compute MLM logits and NSP/SOP logits
  - Loss: MLM cross-entropy (ignore -100) + NSP/SOP cross-entropy
  - Backward pass + Adam optimizer step
- Record loss per epoch for both models
- Same learning rate, batch size, and number of epochs for fair comparison

### 9. Results & Visualization
- Plot MLM loss curves (BERT vs ALBERT) on the same graph
- Plot total loss curves (MLM + NSP/SOP)
- Print final parameter counts side-by-side
- Print a summary table: model, total params, embedding params, encoder params, final MLM loss, final total loss

## Key Functions/Classes

| Component | Description |
|---|---|
| `MultiHeadAttention` | Standard multi-head scaled dot-product attention |
| `TransformerLayer` | One encoder layer: attention + FFN with residual connections |
| `BERTModel` | Standard BERT with per-layer weights, full embedding, NSP head |
| `ALBERTModel` | ALBERT with shared layer weights, factorized embedding, SOP head |
| `build_vocab(corpus)` | Word-level vocabulary from toy corpus |
| `mask_tokens(input_ids, vocab_size)` | MLM masking (15% selected, 80/10/10 split) |
| `BERTDataset` / `ALBERTDataset` | PyTorch Datasets with NSP / SOP labels |
| `train_model(model, dataloader, epochs)` | Training loop returning loss history |

## Data Flow / Shapes

```
Input sentence pair → Tokenize → [CLS] A [SEP] B [SEP]
    → input_ids: [batch, seq_len]
    → segment_ids: [batch, seq_len]
    → mask_tokens → masked_ids, mlm_labels

Embedding:
  BERT:  input_ids [B, L] → nn.Embedding(V, H) → [B, L, H]
  ALBERT: input_ids [B, L] → nn.Embedding(V, E) → [B, L, E] → Linear(E, H) → [B, L, H]

Encoder (per layer):
  [B, L, H] → MultiHeadAttention → [B, L, H] → FFN → [B, L, H]
  BERT: 4 separate layers (4× params)
  ALBERT: 1 layer applied 4 times (1× params)

Heads:
  MLM: [B, L, H] → Linear(H, V) → [B, L, V]  (cross-entropy with mlm_labels)
  NSP: [CLS] [B, H] → Linear(H, 2) → [B, 2]  (BERT)
  SOP: [CLS] [B, H] → Linear(H, 2) → [B, 2]  (ALBERT)
```

## Deliberate Simplifications vs Full Paper

1. **Scale**: The real ALBERT uses 12–24 layers, H=768–4096, V=30,000. This notebook uses 4 layers, H=128, V~500 for demonstration.
2. **Tokenization**: Word-level tokenization instead of WordPiece/SentencePiece. No subword tokens.
3. **Pretraining data**: A toy corpus of ~50 sentences vs Wikipedia + BookCorpus (3.3B words).
4. **Masking**: Static masking (same as BERT) rather than dynamic masking (RoBERTa improvement) — the focus here is on ALBERT's parameter reduction, not masking strategy.
5. **No downstream evaluation**: The paper evaluates on GLUE, RACE, SQuAD. This notebook only compares pretraining loss, not fine-tuned downstream accuracy.
6. **No LAMB optimizer**: The paper uses LAMB for large batch training. This notebook uses standard Adam.
7. **SOP vs NSP comparison**: Both losses are implemented, but the comparison focuses on the auxiliary task difference alongside the parameter reduction, not the SOP vs NSP ablation in isolation (which the paper does separately).
