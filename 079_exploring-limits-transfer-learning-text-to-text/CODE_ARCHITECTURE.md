# Code Architecture — T5 (Text-to-Text Transfer Transformer)

## Overview

This notebook implements the core concepts of the T5 paper from scratch in PyTorch:
1. A small encoder-decoder Transformer with multi-head attention, relative position bias, and shared input/output embeddings
2. The span corruption pre-training objective (mask contiguous spans, predict sentinel-interleaved originals)
3. The text-to-text format applied to multiple toy NLP tasks (sentiment classification, summarization, question answering)
4. Fine-tuning the pre-trained model on downstream tasks using the same loss, same architecture, same everything

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install and import `torch`, `torch.nn`, `torch.nn.functional`, `numpy`, `collections`
- Set device (GPU if available)
- Set random seeds for reproducibility

### 2. Vocabulary & Tokenizer
- **Class `SimpleTokenizer`**: A minimal whitespace + punctuation tokenizer that builds a vocabulary from a corpus.
  - `fit(texts)`: builds vocab with `<pad>`, `<sos>`, `<eos>`, `<unk>`, and sentinel tokens `<extra_id_0>` through `<extra_id_9>`
  - `encode(text)` → list of token IDs
  - `decode(ids)` → text string
  - Properties: `pad_id`, `sos_id`, `eos_id`, `vocab_size`
- **Design choice:** We use a simple word-level tokenizer instead of BPE/SentencePiece to keep the notebook self-contained and easy to understand. The paper uses SentencePiece with a 32K vocabulary.

### 3. Transformer Building Blocks

#### 3a. Multi-Head Attention (`MultiHeadAttention`)
- Input: Q, K, V tensors of shape `(batch, seq_len, d_model)`
- Linear projections to `n_heads` attention heads
- Scaled dot-product attention: `softmax(QK^T / sqrt(d_k)) V`
- **Relative position bias** added to attention logits before softmax: a learned scalar per (relative_distance, head) pair, shared across layers
- Output: concatenated heads → linear projection
- Shapes: `(batch, seq_len, d_model)` → `(batch, seq_len, d_model)`

#### 3b. Position-wise Feed-Forward Network (`PositionwiseFFN`)
- Two linear layers with ReLU activation in between
- `d_model → d_ff → d_model`
- Simplified layer normalization (rescale only, no additive bias) applied to inputs (Pre-LN style)

#### 3c. Encoder Block (`EncoderBlock`)
- Self-attention (bidirectional) → residual + dropout
- Feed-forward → residual + dropout
- Pre-LN: layer norm before each sublayer

#### 3d. Decoder Block (`DecoderBlock`)
- Causal self-attention (masked) → residual + dropout
- Cross-attention to encoder output → residual + dropout
- Feed-forward → residual + dropout

#### 3e. Full Transformer (`T5Model`)
- Token embedding (shared between encoder input, decoder input, and output softmax)
- Stack of N encoder blocks + N decoder blocks
- Final layer norm + linear projection to vocab size
- `forward(src_ids, tgt_ids, src_mask, tgt_mask)` → logits `(batch, tgt_len, vocab_size)`

### 4. Span Corruption Pre-training

#### 4a. Span Corruption Function (`corrupt_spans`)
- Input: token IDs, corruption_rate (default 0.15), mean_span_length (default 3)
- Randomly selects ~15% of tokens to corrupt, grouped into contiguous spans
- Each span is replaced by a unique sentinel token `<extra_id_k>`
- Returns: (corrupted_input_ids, target_ids)
  - target_ids = `<extra_id_0> span_0 <extra_id_1> span_1 ... <eos>`
  - corrupted_input_ids = original with spans replaced by sentinels

#### 4b. Pre-training Data
- A small synthetic corpus of English sentences (generated programmatically)
- For each sentence: tokenize → corrupt spans → create (input, target) pair
- DataLoader collates into padded batches

### 5. Pre-training Loop
- **Loss:** Cross-entropy on decoder output (ignoring pad tokens)
- **Optimizer:** Adam with linear warmup + inverse-square-root decay
- **Epochs:** 20 (small model, small data — enough to show learning)
- **Metrics:** Training loss curve
- The model learns to reconstruct corrupted spans, developing general text representations

### 6. Text-to-Text Downstream Tasks

#### 6a. Task Definitions
Three toy tasks, each formatted as text → text:

1. **Sentiment Classification**
   - Input: `"classify sentiment: This movie was great!"`
   - Target: `"positive"` (or `"negative"`)
   - ~200 toy examples from templates

2. **Summarization**
   - Input: `"summarize: <a paragraph of several sentences>"`
   - Target: `"<one-sentence summary>"`
   - ~100 toy examples (paragraphs built from sentence templates)

3. **Question Answering**
   - Input: `"question: What color is the sky? context: The sky is blue on a clear day."`
   - Target: `"blue"`
   - ~100 toy QA pairs

#### 6b. Multi-task Fine-tuning
- Mix all three tasks into one training set (with task prefix in input)
- Fine-tune the pre-trained model for 30 epochs
- Same loss, same model, same training procedure for all tasks

### 7. Evaluation & Demonstration
- Run inference on held-out examples for each task
- Greedy decoding: feed `<sos>`, generate one token at a time until `<eos>`
- Print predictions vs. ground truth for each task
- Show that a single model handles classification, summarization, and QA

## Key Data Flow / Shapes

```
Input text → tokenizer.encode → token IDs (batch, seq_len)
  → embedding (batch, seq_len, d_model)
  → encoder blocks (batch, seq_len, d_model) — bidirectional
  → decoder:
      target IDs (batch, tgt_len) → embedding
      → causal self-attention + cross-attention to encoder output
      → linear projection (batch, tgt_len, vocab_size)
      → softmax → predicted token probabilities
  → argmax → predicted token IDs → tokenizer.decode → output text
```

## Deliberate Simplifications vs. Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Tokenizer | SentencePiece (32K subword) | Whitespace + punctuation word-level (~500 vocab) |
| Model size | 60M – 11B parameters | ~2M parameters (2 enc + 2 dec layers, 128 d_model, 4 heads) |
| Pre-training data | 750 GB C4 corpus | ~500 synthetic English sentences |
| Pre-training duration | 1M+ steps on 512 TPUs | 20 epochs on GPU in minutes |
| Tasks | 30+ benchmarks (GLUE, SQuAD, CNN/DM, WMT) | 3 toy tasks (~500 examples total) |
| Position embeddings | Relative position bias (buckets) | Relative position bias (linear distance) |
| Optimizer | Adafactor | Adam |
| Beam search decoding | Yes (beam width 4) | Greedy decoding |

These simplifications keep the notebook runnable in under 30 minutes on a single GPU while demonstrating every core concept: the text-to-text format, span corruption pre-training, encoder-decoder architecture, relative position bias, and multi-task fine-tuning with a single model.
