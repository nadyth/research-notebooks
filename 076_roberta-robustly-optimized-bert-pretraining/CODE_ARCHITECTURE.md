# Code Architecture — 076 RoBERTa

## Overview

This notebook implements a from-scratch BERT-style Transformer encoder and pretrains it under two recipes side-by-side:

1. **BERT recipe** — static masking + NSP task + smaller batch
2. **RoBERTa recipe** — dynamic masking + no NSP + larger batch

Both recipes use the same model architecture, vocabulary, and toy corpus. The notebook compares pretraining loss curves to demonstrate that RoBERTa's recipe leads to faster convergence and lower final loss — the core finding of the paper.

The model is intentionally small (2 Transformer layers, 128 hidden dim, 4 attention heads, ~500 vocabulary) to run within Kaggle's 30-minute GPU limit while demonstrating all core RoBERTa mechanisms.

## Section-by-Section Breakdown

### 1. Imports & Configuration
- Imports: `torch`, `torch.nn`, `torch.nn.functional`, `math`, `random`, `collections`, `matplotlib.pyplot`
- Hyperparameters: `d_model=128`, `n_heads=4`, `n_layers=2`, `vocab_size` (built from corpus), `max_len=64`, `lr=1e-4`
- BERT recipe: `batch_size=16`, `mask_strategy='static'`, `use_nsp=True`, `epochs=15`
- RoBERTa recipe: `batch_size=64`, `mask_strategy='dynamic'`, `use_nsp=False`, `epochs=15`

### 2. Toy Corpus Construction
- A small collection of English sentences organised into short documents (2-4 sentences each)
- For NSP training (BERT recipe): each training example is a sentence pair (A, B) where B is either the actual next sentence or a random one
- For no-NSP training (RoBERTa recipe): each training example is a single sentence or a contiguous document chunk
- Word-level tokenizer (split on whitespace, lowercase) — deliberately simpler than BPE for clarity

### 3. Vocabulary Building
- Build vocabulary from all words in the toy corpus
- Special tokens: `[PAD]=0`, `[UNK]=1`, `[CLS]=2`, `[SEP]=3`, `<tool_call>MASK]=4`
- Word-to-id and id-to-word mappings (~500 tokens)

### 4. Masking Functions

#### 4a. Static Masking (BERT recipe)
- Pre-compute masked positions once at dataset creation time
- The same positions remain masked for every epoch — the model always sees the same puzzle
- Returns: masked_input_ids, mlm_labels (original token at masked positions, -100 elsewhere)

#### 4b. Dynamic Masking (RoBERTa recipe)
- Re-sample 15% of positions at each call (every epoch gets fresh masks)
- Of selected positions: 80% → `<tool_call>MASK]`, 10% → random token, 10% → unchanged
- Returns: masked_input_ids, mlm_labels
- This is the key RoBERTa innovation — effectively data augmentation

### 5. Dataset Construction

#### 5a. BERT Dataset (with NSP)
- Each example: `[CLS] A [SEP] B [SEP]` with segment IDs (0 for A, 1 for B)
- NSP label: 1 if B follows A, 0 if random
- MLM labels computed with static masking
- Returns: (input_ids, segment_ids, attention_mask, mlm_labels, nsp_label)

#### 5b. RoBERTa Dataset (no NSP)
- Each example: `[CLS] sentence [SEP]` (single sentence) or `[CLS] doc_chunk [SEP]`
- No segment IDs needed (all 0), no NSP label
- MLM labels computed with dynamic masking at each `__getitem__` call
- Returns: (input_ids, segment_ids, attention_mask, mlm_labels)

### 6. Model Architecture

#### 6a. Multi-Head Self-Attention
- Standard scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
- Multi-head: split `d_model` into `n_heads` heads, attend independently, concat, project
- Fully bidirectional (no causal mask) — same as BERT

#### 6b. Position-wise Feed-Forward
- `FFN(x) = GELU(xW1 + b1)W2 + b2` with inner dimension `d_model * 4`

#### 6c. Transformer Encoder Layer
- Post-LN: `LayerNorm(x + MultiHeadAttn(x))`, then `LayerNorm(x + FFN(x))`

#### 6d. BERT/RoBERTa Model
- **Embedding:** `token_emb + segment_emb + position_emb` (all learned)
- **Encoder:** stack of `n_layers` Transformer encoder layers
- **MLM head:** Linear → GELU → LayerNorm → output projection over vocabulary
- **NSP head** (BERT recipe only): Linear(2) on `[CLS]` token output
- The architecture is identical for both recipes — only the training recipe differs

### 7. Training Loops

#### 7a. BERT Recipe Training
- Loss = MLM loss + NSP loss (both cross-entropy)
- Static masking (same masks every epoch)
- Smaller batch size (16)
- Track and record average MLM loss per epoch

#### 7b. RoBERTa Recipe Training
- Loss = MLM loss only (no NSP)
- Dynamic masking (re-mask every epoch)
- Larger batch size (64)
- Track and record average MLM loss per epoch

### 8. Comparison & Visualization
- Plot MLM loss curves for both recipes on the same chart
- Print final loss values and percentage improvement
- Show sample MLM predictions from each model
- The key finding: RoBERTa recipe converges faster and to lower loss, demonstrating the paper's thesis

## Key Functions/Classes

| Name | Purpose |
|------|---------|
| `MultiHeadAttention` | Standard multi-head self-attention |
| `TransformerEncoderLayer` | One layer of the encoder |
| `BertModel` | Full BERT/RoBERTa encoder with MLM + optional NSP heads |
| `static_mask()` | BERT-style fixed masking |
| `dynamic_mask()` | RoBERTa-style per-epoch fresh masking |
| `BertDataset` | Dataset with NSP + static masking |
| `RobertaDataset` | Dataset without NSP + dynamic masking |
| `train_bert_recipe()` | Pretraining with BERT recipe |
| `train_roberta_recipe()` | Pretraining with RoBERTa recipe |

## Data Flow / Shapes

```
Input sentence → tokenize → [CLS] tokens [SEP] → input_ids [batch, seq_len]
                                                    ↓
                                         Embedding [batch, seq_len, d_model]
                                                    ↓
                                         Encoder (n_layers ×)
                                                    ↓
                                         Hidden states [batch, seq_len, d_model]
                                                    ↓
                                    MLM head → logits [batch, seq_len, vocab_size]
                                    NSP head (BERT only) → logits [batch, 2]
```

## Deliberate Simplifications vs Full Paper

1. **Vocabulary:** Word-level (~500 tokens) vs byte-level BPE (50k). Keeps the implementation simple and understandable.
2. **Corpus size:** ~200 sentences vs 160GB of text. The comparison of recipes is still meaningful at toy scale.
3. **Model size:** 2 layers / 128 dim vs 24 layers / 1024 dim. Architecture is identical in structure, just smaller.
4. **Batch size scaling:** 16 vs 64 (4×) rather than 256 vs 8192 (32×). The relative comparison still holds.
5. **No fine-tuning:** The paper fine-tunes on GLUE/RACE/SQuAD; we focus on pretraining loss comparison only, which is the paper's primary controlled experiment.
6. **No byte-level encoding:** We use word-level tokenisation. Byte-level BPE is an engineering choice, not a research contribution.
