# Code Architecture — 075 XLNet

## Overview

This notebook implements the **permutation language modeling (PLM)** objective from XLNet on a small Transformer, and compares it against a standard **masked language model (MLM)** baseline (BERT-style) on the same toy corpus. The implementation demonstrates the core mechanism of XLNet — predicting tokens in random factorization orders using two-stream self-attention — at a scale that runs within Kaggle's GPU time limit.

The notebook is organized into these phases:
1. Toy corpus construction and vocabulary building
2. A shared small Transformer encoder
3. MLM baseline implementation (standard masked LM)
4. Permutation LM implementation with two-stream attention
5. Training both models and comparing perplexity / loss

## Section-by-Section Breakdown

### 1. Imports & Configuration
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `math`, `random`, `itertools`, `collections`, `matplotlib`
- Hyperparameters: `d_model=128`, `n_heads=4`, `n_layers=3`, `vocab_size` (built from corpus), `max_len=32`, `batch_size=16`, `lr=5e-4`, `epochs=20`, `n_perm_samples=6` (number of permutations sampled per sequence per epoch), `partial_pred_ratio=0.5` (fraction of positions used for loss)

### 2. Toy Corpus Construction
- A small corpus of ~100 simple English sentences (3-12 words each)
- Word-level tokenization (split on whitespace, lowercase)
- Vocabulary built from all words + special tokens: `[PAD]=0`, `[UNK]=1`, `[CLS]=2`, `[SEP]=3`
- ~200-400 vocabulary tokens

### 3. Shared Transformer Components

#### 3a. Multi-Head Self-Attention with Custom Masking
- Standard scaled dot-product attention: `softmax(QK^T / sqrt(d_k) + rel_pos_bias) V`
- Supports arbitrary attention masks (critical for permutation-based masking)
- Multi-head: split `d_model` into `n_heads` heads, attend independently, concat, project

#### 3b. Position-wise Feed-Forward
- `FFN(x) = GELU(xW1 + b1)W2 + b2` with inner dimension `d_model * 4`

#### 3c. Transformer Layer (parameter-shared for both streams)
- Post-LN: `LayerNorm(x + SubLayer(x))`
- Used for both content stream and query stream layers

### 4. MLM Baseline (BERT-style)
- **Masking:** 15% of tokens selected; of those 80% → `[MASK]`, 10% → random, 10% → unchanged
- **Attention:** Fully bidirectional (no causal mask) — every token sees every other token
- **Loss:** CrossEntropy at masked positions only (ignore_index=-100 for unmasked)
- **Architecture:** Standard Transformer encoder → linear projection to vocab size
- This is the comparison baseline

### 5. Permutation Language Model (XLNet-style)

#### 5a. Permutation Generation
- For each input sequence of length L, sample `n_perm_samples` random permutations of [0, 1, ..., L-1]
- Each permutation defines a factorization order for AR prediction
- Permutations are applied via attention masks, NOT by reordering the input sequence itself

#### 5b. Two-Stream Self-Attention
- **Content stream:** h_t = Attention(query=h_t, key=h_{<=t in perm}, value=h_{<=t in perm}) — includes current token's embedding
- **Query stream:** g_t = Attention(query=g_t, key=h_{<t in perm}, value=h_{<t in perm}) — excludes current token's content
- Both streams use the same Transformer parameters (shared weights)
- Attention mask is derived from the permutation: position j can attend to position i if i appears before j in the permutation

#### 5c. Attention Mask Construction
- Given a permutation z = [z_0, z_1, ..., z_{L-1}], the mask for the content stream allows position z_t to attend to positions {z_0, ..., z_t} (inclusive — it sees its own content)
- The mask for the query stream allows position z_t to attend to positions {z_0, ..., z_{t-1}} (exclusive — it cannot see its own content)
- Masks are precomputed as [L, L] boolean matrices for each permutation

#### 5d. Partial Prediction
- Only the last `partial_pred_ratio` fraction of positions in the permutation are used for loss
- E.g., for a length-6 sequence with ratio 0.5, only positions z_3, z_4, z_5 generate loss
- This focuses training on positions with richer context (more preceding tokens in the permutation)

#### 5e. Forward Pass (Permutation LM)
1. Embed all tokens: `emb = token_emb(input_ids) + pos_emb(positions)`
2. Content stream: h = Transformer(emb, mask=content_mask) — processes all positions with their own content
3. Query stream: g = Transformer(query_emb, mask=query_mask, kv_from=h) — predicts each position without seeing its own content
4. For target positions: logits = Linear(g_z_t), loss = CrossEntropy(logits, target_token)

### 6. Training Loop
- For each epoch:
  - Train MLM baseline: forward → MLM loss → backprop → Adam step
  - Train PLM model: for each batch, sample permutations → forward → PLM loss → backprop → Adam step
  - Record both losses
- Print loss every 5 epochs; plot both loss curves at end

### 7. Evaluation & Comparison
- Compare final training loss / perplexity of MLM vs PLM
- Generate sample predictions: given a partial sentence, show what each model predicts for held-out tokens
- Plot loss curves side by side
- Print a comparison table: final loss, perplexity, and sample predictions

## Key Functions/Classes

| Name | Type | Purpose |
|---|---|---|
| `MultiHeadAttention` | Module | Scaled dot-product attention with custom mask support |
| `TransformerLayer` | Module | One encoder layer (attn + FFN + residual + LN) |
| `MLMModel` | Module | Standard masked LM: embeddings → encoder → vocab projection |
| `PermutationLM` | Module | XLNet PLM: two-stream attention with permutation masks |
| `generate_permutations` | Function | Sample n random permutations of [0, L-1] |
| `build_attention_masks` | Function | Build content and query stream masks from a permutation |
| `apply_mlm_masking` | Function | Apply 80/10/10 BERT-style masking to a sequence |
| `ToyCorpus` | Class | Manages the toy corpus and vocabulary |
| `train_mlm` | Function | Training loop for MLM baseline |
| `train_plm` | Function | Training loop for permutation LM |
| `compute_perplexity` | Function | Compute perplexity from average loss |

## Data Flow / Shapes

```
Input sentence → tokenize → [CLS] tok1 tok2 ... tokL [SEP]
→ input_ids: [batch, seq_len] (int)
→ token_embeddings: [batch, seq_len, 128]

MLM path:
→ mask 15% tokens → [batch, seq_len] with some [MASK]
→ encoder (bidirectional) → [batch, seq_len, 128]
→ Linear → [batch, seq_len, vocab_size]
→ CrossEntropy at masked positions

PLM path:
→ sample permutation z: [seq_len] (e.g., [3, 1, 4, 0, 2, 5])
→ build content_mask: [seq_len, seq_len] (position z_t sees z_{<=t})
→ build query_mask: [seq_len, seq_len] (position z_t sees z_{<t})
→ content stream: h = Transformer(emb, content_mask) → [batch, seq_len, 128]
→ query stream: g = Transformer(query_emb, query_mask, kv=h) → [batch, seq_len, 128]
→ Linear(g) → [batch, seq_len, vocab_size]
→ CrossEntropy at partial-prediction target positions
```

## Deliberate Simplifications vs Full Paper

| Simplification | Original XLNet | Our Version | Rationale |
|---|---|---|---|
| Model size | 12-24 layers, 768-1024 hidden | 3 layers, 128 hidden | Runs in <15 min on T4 GPU |
| Permutations | All n! (approximated by sampling) | 6 random samples per sequence | Demonstrates mechanism without exponential cost |
| Two-stream attention | Separate content & query streams with shared params | Same, simplified to single-mask approach | Faithful but pedagogically clearer |
| Transformer-XL memory | Segment-level recurrence with memory cache | Omitted (fixed-length sequences) | Toy corpus has short sentences; memory not needed |
| Relative positional encoding | Full relative position bias matrix | Learned absolute position embeddings | Simpler; adequate for short sequences |
| Partial prediction | Last 1/K tokens (K=6 in paper) | Last 50% of permutation positions | Higher ratio ensures enough loss signal on small data |
| Vocabulary | SentencePiece, 32k subwords | ~200-400 word-level | No tokeniser dependency |
| Pre-training data | BooksCorpus + Wikipedia (126GB) | ~100 toy sentences | Demonstrate objective, not scale |
| Fine-tuning | 20 downstream tasks | None (objective comparison only) | Focus on the PLM vs MLM comparison |
| Target-aware representations | Relative encoding with two-stream | Simplified two-stream with shared encoder | Core mechanism preserved |
