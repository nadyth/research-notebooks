# Code Architecture — Tacotron: End-to-End Speech Synthesis

## Notebook Structure

The notebook implements a simplified Tacotron from scratch in PyTorch, training a character-to-mel-spectrogram seq2seq model with attention on a toy dataset, and visualizing the predicted spectrogram and attention alignment.

### Section 1: Setup and Imports
- PyTorch, NumPy, matplotlib, librosa (for mel-spectrogram extraction and Griffin-Lim)
- All packages pip-installable; no pretrained models

### Section 2: Character Embedding and Text Processing
- `char_to_idx(text)`: Maps each character to an integer index, builds a vocabulary from a small character set (a-z, space, punctuation, digits)
- `idx_to_embed(indices, vocab_size, embed_dim)`: Embeds character indices into 256-D vectors (matches paper's 256-D character embedding)
- **Shapes**: Input: string → indices (T_text,) → embeddings (T_text, 256)

### Section 3: Pre-Net Module
- `PreNet(embed_dim, hidden_dims=[256, 128], dropout=0.5)`: Two FC layers with ReLU and 50% dropout
- Applied to both encoder input (character embeddings) and decoder input (mel frames)
- The bottleneck dropout is critical — it provides noise that helps the model generalize to the multi-modal output distribution
- **Shapes**: Input (B, T, 256) → Output (B, T, 128)

### Section 4: CBHG Module
- `CBHG(K=16, channels=128, highway_layers=4, gru_cells=128)`:
  - **Conv1D bank**: K sets of 1-D convolutions, the k-th set has width k (1, 2, ..., K). Each set has 128 output channels with ReLU. Outputs are stacked along the channel dimension. This captures n-gram-like local context (unigrams through K-grams).
  - **Max pooling**: stride=1, width=2 — preserves time resolution while adding local invariance
  - **Conv1D projections**: conv-3-128-ReLU → conv-3-128-Linear with residual connection (added to input)
  - **Highway network**: 4 layers of FC-128-ReLU with transform and carry gates, enabling information flow across layers
  - **Bidirectional GRU**: 128 cells each direction, captures forward and backward sequential context
- **Shapes**: Input (B, T, 128) → conv bank (B, T, K*128) → maxpool (B, T, K*128) → proj (B, T, 128) → +residual → highway (B, T, 128) → BiGRU (B, T, 256)

### Section 5: Encoder
- `Encoder(vocab_size, embed_dim, prenet_dims, cbhg_config)`: Character embedding → PreNet → CBHG
- Produces the encoder output that the attention module attends over
- **Shapes**: Input: (B, T_text) indices → Output: (B, T_text, 256)

### Section 6: Attention Decoder
- `AttentionDecoder(encoder_dim=256, decoder_dim=256, attention_dim=256, mel_dim=80, r=2)`:
  - **Attention RNN**: 1-layer GRU (256 cells) producing the attention query at each decoder step
  - **Content-based tanh attention**: score = tanh(W_query * query + W_key * encoder_output), then softmax over encoder timesteps to get attention weights, context vector = weighted sum of encoder outputs
  - **Decoder RNNs**: 2-layer residual GRU stack (256 cells), input is [context_vector; attention_rnn_output]
  - **Output layer**: FC → predicts r × mel_dim frames (80 × 2 = 160 values) per decoder step
  - **Reduction factor r=2**: Each decoder step emits 2 mel frames, reducing total decoder steps by half. This is crucial for fast and stable alignment learning.
  - **Teacher forcing**: During training, every r-th ground truth mel frame is fed as input to the next decoder step. An all-zero GO frame starts decoding.
- **Shapes**: 
  - Input: encoder output (B, T_text, 256), GO frame (B, 1, 80)
  - Per step: attention query (B, 256) → context (B, 256) → decoder output (B, 160) → reshaped to (B, 2, 80)

### Section 7: Post-Processing Net
- `PostNet(mel_dim=80, linear_dim=80, cbhg_config)`: CBHG (K=8) that sees the full decoded mel sequence and refines it
  - Input: concatenated mel frames from decoder (B, T_mel, 80)
  - CBHG processes the entire sequence bidirectionally — can correct errors using future context
  - Output projections map to linear-frequency spectrogram dimension
- **Deliberate simplification**: The paper predicts linear spectrogram (separate from mel) for Griffin-Lim. Our simplified version focuses on mel prediction and visualization, using Griffin-Lim directly on mel for waveform synthesis.

### Section 8: Full Tacotron Model
- `Tacotron(vocab_size, ...)`:
  - Encoder → AttentionDecoder → PostNet
  - Forward pass: text → mel predictions + linear predictions
  - **Loss**: L1 loss on mel-spectrogram (seq2seq) + L1 loss on linear spectrogram (post-net), equal weights
  - The model also reconstructs zero-padded frames (no loss masking) so it learns when to stop emitting

### Section 9: Toy Dataset
- Synthetic dataset: Generate simple audio signals (e.g., sine wave patterns with text-like character sequences)
- Each sample: a short character sequence paired with a corresponding mel-spectrogram
- The goal is to demonstrate the architecture and attention learning, not produce natural speech
- **Shapes**: Batch of (text_indices, mel_spectrogram) pairs, mel shape (T_mel, 80)

### Section 10: Training Loop
- Adam optimizer, learning rate 0.001 with step decay
- Batch size 8 (reduced from paper's 32 for toy dataset)
- Train for ~200 epochs (enough for attention to converge on toy data)
- Track and visualize attention alignment every 50 epochs
- **Deliberate simplification**: Far fewer steps than the paper's millions of steps on 24.6 hours of real speech

### Section 11: Visualization and Inference
- **Attention alignment plot**: Show the attention weight matrix (T_decoder × T_encoder) to verify the model learned a monotonic alignment
- **Mel-spectrogram comparison**: Plot predicted vs. ground-truth mel-spectrograms side by side
- **Griffin-Lim synthesis**: Convert predicted mel-spectrogram to audio waveform using Griffin-Lim (50 iterations)
- **Waveform plot**: Show the synthesized waveform

## Deliberate Simplifications vs. Full Paper

1. **Toy dataset instead of real speech**: We use synthetic mel-spectrograms paired with character sequences rather than 24.6 hours of professional speech. The goal is architectural demonstration, not natural speech quality.
2. **Reduced CBHG K**: Encoder CBHG uses K=8 instead of K=16; post-net uses K=4 instead of K=8 — sufficient for the toy task, faster to train.
3. **Fewer training steps**: ~200 epochs vs. millions of steps. Real convergence needs much more data and training.
4. **Simplified post-processing**: The post-net maps mel to mel refinement rather than mel to linear spectrogram. Griffin-Lim is applied directly to mel.
5. **No text normalization**: The paper handles abbreviations, numbers, etc. Our toy text is already simple.
6. **Batch size 8 instead of 32**: Fits comfortably in GPU memory for the small model.
7. **No learning rate schedule details**: Simple step decay instead of the paper's multi-step schedule.
