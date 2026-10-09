# Code Architecture — ELMo (073)

## Overview

This notebook implements a simplified ELMo (Embeddings from Language Models) from scratch in PyTorch. The implementation follows the core ideas of Peters et al. (2018): a bidirectional LSTM language model is pre-trained on a toy corpus, then layer-wise contextual embeddings are extracted and shown to vary with context.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Imports: `torch`, `torch.nn`, `torch.optim`, `numpy`, `collections`, `itertools`, `matplotlib`.
- Sets random seed for reproducibility.
- Device auto-detection (GPU if available, else CPU).

### 2. Toy Corpus & Vocabulary
- A small collection of sentences containing words with multiple senses (e.g., "bank" in financial vs. river contexts, "play" as verb vs. noun, "right" as direction vs. correctness).
- `build_vocab()`: maps each unique word to an integer index. Creates `word2idx` and `idx2word` dictionaries.
- Vocabulary size is intentionally small (~50-80 words) so the model trains in minutes.

### 3. BiLM Architecture (`BiLM` class)
- **Embedding layer** (`nn.Embedding`): maps word indices to dense vectors (dim=128). This is the "layer 0" (non-contextual) representation.
- **Forward LSTM** (`nn.LSTM`): multi-layer (2 layers), unidirectional, hidden dim=256. Processes the sentence left-to-right, predicting the next token.
- **Backward LSTM** (`nn.LSTM`): multi-layer (2 layers), unidirectional, hidden dim=256. Processes the sentence right-to-left, predicting the previous token.
- **Output projections** (`nn.Linear`): two linear layers that project LSTM hidden states to vocabulary size for computing LM loss.
- `forward(sentence)`: runs both forward and backward passes, returns hidden states from all layers plus LM logits.

**Key shapes:**
- Input: `(seq_len, batch=1)` — word indices
- Embedding: `(seq_len, 1, emb_dim=128)`
- Each LSTM layer hidden: `(seq_len, 1, hidden_dim=256)`
- Forward+backward concatenation per layer: `(seq_len, 1, 512)`

### 4. Pre-training Loop
- Loss = forward LM loss + backward LM loss.
  - Forward: predict token *t+1* given tokens *0..t* using the forward LSTM.
  - Backward: predict token *t-1* given tokens *t..N* using the backward LSTM.
- Cross-entropy loss, Adam optimizer, lr=0.001.
- Trains for ~50 epochs on the toy corpus.
- Prints loss every 10 epochs.

### 5. ELMo Embedding Extraction (`get_elmo_embeddings` function)
- After pre-training, runs the biLM on input sentences in eval mode (no gradient).
- For each token, collects the hidden state from:
  - Layer 0: the input embedding (non-contextual)
  - Layer 1: concatenation of forward LSTM layer-1 + backward LSTM layer-1
  - Layer 2: concatenation of forward LSTM layer-2 + backward LSTM layer-2
- Returns a list of per-layer embeddings for each token.

### 6. Context Comparison Visualization
- Selects target words that appear in multiple sentences with different meanings (e.g., "bank", "play", "right").
- For each target word, extracts ELMo embeddings from each sentence context.
- Computes cosine similarity between:
  - **Layer 0 (static embedding):** same word in different contexts → high similarity (expected, since it's non-contextual)
  - **Layer 1 (syntax-level):** moderate context sensitivity
  - **Layer 2 (semantics-level):** lower similarity across different senses (expected, since higher layers capture meaning)
- Plots a heatmap or bar chart showing how similarity decreases at higher layers for polysemous words, and stays high for the same word in similar contexts.

### 7. Learned Layer Weights Demo
- Initializes learnable scalar weights `s_0, s_1, s_2` (softmax-normalized) and a scaling factor `gamma`.
- Demonstrates a simple downstream task: given a small set of labeled sentences, fine-tune the layer weights to maximize classification accuracy.
- Shows that different tasks prefer different layers (e.g., a POS-like task prefers lower layers, a sentiment-like task prefers higher layers).

## Key Functions/Classes

| Name | Type | Purpose |
|---|---|---|
| `build_vocab` | function | Creates word↔index mappings from the corpus |
| `BiLM` | nn.Module | Bidirectional LSTM language model with multi-layer hidden states |
| `train_bilm` | function | Pre-training loop: forward + backward LM loss |
| `get_elmo_embeddings` | function | Extracts layer-wise contextual embeddings for a sentence |
| `cosine_sim` | function | Computes cosine similarity between two embedding vectors |
| `plot_context_comparison` | function | Visualizes how the same word's embedding changes across contexts |
| `LayerWeightDemo` | class | Demonstrates learnable layer-weight combination for a toy task |

## Deliberate Simplifications vs. Full Paper

1. **Character-level CNN:** The original ELMo uses a character-level CNN (2048 char n-gram filters) for the token embedding layer to handle OOV words. We use a simple word-level `nn.Embedding` for clarity.
2. **Corpus size:** Original uses 1B Word Benchmark (~800M tokens). We use ~30 hand-crafted sentences.
3. **Model dimensions:** Original uses 4096 hidden units, 2 biLM layers. We use 256 hidden units, 2 layers.
4. **ELMo weighting:** Original learns layer weights via backprop during downstream fine-tuning with the biLM frozen. We demonstrate the same concept on a toy task.
5. **Dropout:** Original applies dropout to LSTM layers. We omit it since our corpus is tiny and overfitting to it is acceptable for demonstration.
6. **No downstream benchmark tasks:** The original evaluates on SQuAD, SNLI, SRL, etc. We only show context-sensitivity of embeddings.
