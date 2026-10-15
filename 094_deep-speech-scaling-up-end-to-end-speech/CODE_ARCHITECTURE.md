# Code Architecture — Deep Speech

## Overview

The notebook implements a from-scratch Deep Speech model: a 5-layer RNN trained with CTC loss on audio spectrograms, producing character-level transcriptions. The implementation uses PyTorch and a synthetic toy dataset (generated programmatically) so it runs entirely self-contained on Kaggle/Colab without downloading large speech corpora.

## Section-by-Section Breakdown

### 1. Imports and Setup
- `torch`, `torch.nn`, `torchaudio` for model and audio processing
- `librosa` for spectrogram computation (log filter banks)
- `numpy`, `matplotlib` for data handling and visualization
- Sets device (CUDA if available), random seeds for reproducibility

### 2. Character Vocabulary
- Alphabet: 29 characters — a–z (26), space, apostrophe, blank (CTC blank token)
- `char_to_idx` and `idx_to_char` mappings
- Blank token is index 0 (standard for PyTorch CTC)

### 3. Synthetic Speech Data Generation
- **Deliberate simplification**: Instead of using real speech corpora (Switchboard/Fisher require LDC access), we synthesize simple "audio" signals that encode character sequences as tone patterns
- Each "utterance" is a sequence of tones where each character maps to a distinct frequency pattern over a fixed number of spectrogram frames
- This gives us labeled (audio, text) pairs for training without external data downloads
- `generate_utterance(text)`: converts text → spectrogram frames
- `SyntheticSpeechDataset`: PyTorch Dataset wrapping generated utterances
- Each character gets N frames of a unique frequency signature; this is learnable by the RNN but deliberately simple

### 4. Spectrogram Features
- **Feature extraction**: Compute log filter bank features from the synthetic audio
- For the toy implementation, features are generated directly as spectrogram-like vectors
- Feature dimension: 80 (matching paper's 80 filter banks for Switchboard)
- Normalization: per-utterance mean subtraction and variance normalization

### 5. Data Loading and Batching
- Collate function handles variable-length sequences: sorts by length, pads with zeros
- Returns `(padded_spectrograms, input_lengths, target_labels, target_lengths)`
- Batch size: 16–32 (small for toy dataset)

### 6. Deep Speech Model (`DeepSpeechModel`)
- **Layer 1 (Feed-forward with context)**: Conv1d that processes ±9 frames of context. Implemented as a 1D convolution with kernel_size=19 over the feature dimension, or equivalently a linear layer on concatenated context frames. Output: 256 hidden units.
- **Layers 2–3 (Feed-forward)**: Linear layers with clipped ReLU, 256 units each
- **Layer 4 (Bidirectional RNN)**: `nn.LSTM` or `nn.RNN` with bidirectional=True, hidden_size=256. Forward and backward hidden states are summed (not concatenated) per the paper.
- **Layer 5 (Feed-forward)**: Linear layer, 256 units
- **Output layer**: Linear → softmax over 29 characters
- **Clipped ReLU**: Custom activation `g(z) = min(max(0, z), 20)` applied to all hidden layers
- **Dropout**: 5–10% on feed-forward layers

### 7. CTC Loss
- Uses `torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)`
- Inputs: log-probabilities from the model (T, B, C), target labels, input lengths, target lengths
- CTC automatically handles the alignment between variable-length audio frames and character sequences

### 8. Training Loop
- **Optimizer**: Nesterov Accelerated Gradient (SGD with momentum=0.99, nesterov=True)
- **Learning rate schedule**: Anneal by constant factor after each epoch
- **Epochs**: 30 (small dataset converges quickly)
- **Gradient clipping**: Max norm 5.0 to prevent exploding gradients
- Tracks and plots training loss per epoch

### 9. Greedy CTC Decoder
- `greedy_ctc_decode(log_probs)`: Takes argmax at each timestep, collapses repeated characters, removes blank tokens
- Returns decoded character string

### 10. Beam Search Decoder (Simplified)
- `beam_search_decode(log_probs, beam_width=100)`: Implements a basic beam search over CTC output probabilities
- Without language model (simplified vs paper's full N-gram beam search with α/β tuning)
- Returns top beam decoded string

### 11. Evaluation
- Computes Character Error Rate (CER) and Word Error Rate (WER) on test set
- CER = Levenshtein distance at character level / reference length
- WER = Levenshtein distance at word level / reference length
- Shows sample predictions vs ground truth

### 12. Visualization
- Plots training loss curve
- Displays sample spectrograms
- Shows prediction examples with CER/WER

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `DeepSpeechModel` | 5-layer RNN: 3 FF + 1 BiRNN + 1 FF + softmax |
| `clipped_relu(z, max_val=20)` | Clipped ReLU activation g(z) = min(max(0,z), 20) |
| `SyntheticSpeechDataset` | Generates toy (spectrogram, text) pairs |
| `collate_fn(batch)` | Pads variable-length sequences, returns batched tensors |
| `greedy_ctc_decode(log_probs)` | Argmax + collapse repeats + remove blanks |
| `beam_search_decode(log_probs, beam_width)` | Simplified beam search CTC decoding |
| `levenshtein(a, b)` | Edit distance for CER/WER computation |
| `cer_rate(pred, ref)` | Character Error Rate |
| `wer_rate(pred, ref)` | Word Error Rate |

## Data Flow / Shapes

```
Audio text → generate_utterance → spectrogram [T, 80]
    ↓
Batch collation → padded batch [B, T_max, 80] + lengths
    ↓
Context conv (Layer 1) → [B, T, 256]
    ↓
FF Layer 2 → [B, T, 256]
    ↓
FF Layer 3 → [B, T, 256]
    ↓
BiRNN Layer 4 → [B, T, 512] → sum directions → [B, T, 256]
    ↓
FF Layer 5 → [B, T, 256]
    ↓
Output Linear → [B, T, 29] → log_softmax → [T, B, 29]
    ↓
CTC Loss(targets, input_lengths, target_lengths) → scalar
    ↓
Greedy/Beam decode → predicted text string
```

## Deliberate Simplifications vs Full Paper

1. **Synthetic data**: The paper uses 5000+ hours of real speech (Switchboard, Fisher, Baidu internal). We generate toy audio programmatically — no external downloads needed.
2. **Model size**: Paper uses 2048–2560 hidden units per layer; we use 256 for fast training on GPU within the 30-minute Kaggle limit.
3. **No language model**: The paper integrates a 4–5 gram KenLM language model via beam search. We use plain CTC greedy/beam decoding without external LM.
4. **No multi-GPU**: The paper uses data + model parallelism across GPUs. We train on a single GPU.
5. **No data synthesis (noise)**: The paper creates noisy training data by superimposing noise tracks. Our toy dataset is clean.
6. **LSTM instead of vanilla RNN**: Paper uses simple ReLU recurrent units; we use LSTM for training stability in the toy setting (note: paper deliberately avoids LSTM for GPU efficiency, but our toy model benefits from LSTM's gating).
7. **No striding**: Paper uses stride-2 in the first layer to reduce sequence length. We skip this optimization since our sequences are already short.
