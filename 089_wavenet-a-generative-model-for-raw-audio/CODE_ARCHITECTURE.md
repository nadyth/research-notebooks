# Code Architecture — WaveNet

## Notebook Structure

The notebook implements a simplified WaveNet from scratch in PyTorch, training on a toy audio signal (a sum of sine waves) and generating new audio samples autoregressively.

### Section 1: Setup and Imports
- PyTorch, NumPy, matplotlib, scipy.io.wavfile for audio I/O
- No pretrained models; everything from scratch

### Section 2: Mu-Law Companding and Quantization
- `mulaw_encode(x, mu=255)`: Applies mu-law companding transformation to float audio in [-1, 1], then quantizes to 256 integer categories (0-255)
- `mulaw_decode(x, mu=255)`: Inverse transform — dequantizes and applies inverse mu-law to reconstruct float audio
- **Why**: Non-linear quantization preserves quiet sounds much better than linear quantization. The paper found this critical for speech quality.
- **Shapes**: Input: (T,) float64 → Output: (T,) int64 in [0, 255]

### Section 3: Toy Audio Dataset
- Generate a synthetic audio signal: sum of sine waves at different frequencies (e.g., 220 Hz + 330 Hz at 8000 Hz sample rate)
- This replaces real speech data — the goal is to demonstrate the architecture, not produce natural speech
- Convert to mu-law quantized categories
- Create training samples by sliding a window over the signal

### Section 4: Causal Dilated Convolution Layer
- `CausalConv1d`: A wrapper around `nn.Conv1d` that pads the input on the left to ensure causality (no future information leaks). Padding = (kernel_size - 1) * dilation.
- After convolution, trim the right padding so output length = input length.

### Section 5: WaveNet Residual Block
- `WaveNetBlock(dilation, channels, kernel_size)`: A single dilated causal conv block with:
  - **Filter branch**: CausalConv1d → tanh
  - **Gate branch**: CausalConv1d → sigmoid
  - **Gated activation**: z = tanh(W_f * x) ⊙ sigmoid(W_g * x)
  - **1x1 conv for residual output** (added back to input)
  - **1x1 conv for skip output** (collected across all layers)
- **Data flow**: Input (B, C, T) → filter (B, C, T), gate (B, C, T) → z (B, C, T) → residual (B, C, T), skip (B, C, T)

### Section 6: Full WaveNet Model
- `WaveNet(layers, stacks, channels, kernel_size, n_classes=256)`:
  - **Input**: Quantized audio categories (B, T) → one-hot encoded (B, 256, T)
  - **1x1 conv**: Maps one-hot to channel dimension (B, C, T)
  - **Stacked dilated blocks**: Dilation pattern [1, 2, 4, ..., 2^(layers-1)] repeated `stacks` times
  - **Skip connections**: Sum all skip outputs → ReLU → 1x1 conv → ReLU → 1x1 conv → output (B, 256, T)
  - **Output**: Softmax distribution over 256 categories at each timestep
- **Receptive field**: 2^(layers) * stacks (with kernel_size=2); e.g., 10 layers × 1 stack = 1024

### Section 7: Training Loop
- **Loss**: Cross-entropy loss over 256 categories, comparing predicted distribution at timestep t to ground truth at timestep t+1 (shifted by one)
- **Optimizer**: Adam with learning rate 0.001
- **Teacher forcing**: All timesteps predicted in parallel — input is the full sequence, target is the sequence shifted left by 1
- **Epochs**: 20-30 on toy data (reduced from paper's full training for time constraints)
- **Batch**: Single long sequence, or batched windows

### Section 8: Generation (Autoregressive Sampling)
- Start with a seed of zeros (or a small chunk of real audio)
- For each new sample:
  1. Feed current sequence through the model
  2. Take the softmax distribution at the last timestep
  3. Sample from the distribution (or take argmax)
  4. Append the sampled category to the sequence
  5. Repeat
- This is slow (O(T^2) in sequence length) but demonstrates the autoregressive nature
- Generate ~2000 samples and decode from mu-law back to audio

### Section 9: Visualization and Audio Output
- Plot the original vs. generated waveform
- Save generated audio as a .wav file
- Plot the training loss curve

## Key Functions/Classes

| Component | Purpose |
|---|---|
| `mulaw_encode/decode` | Non-linear 256-category quantization of audio |
| `CausalConv1d` | Left-padded conv ensuring no future leakage |
| `WaveNetBlock` | Gated dilated conv with residual + skip outputs |
| `WaveNet` | Full stacked model with skip aggregation and softmax head |
| `train()` | Parallel teacher-forced training with cross-entropy |
| `generate()` | Sequential autoregressive sampling |

## Data Flow / Shapes

```
Audio (T,) float
  → mulaw_encode → (T,) int [0,255]
  → one-hot → (1, 256, T)
  → 1x1 conv → (1, C, T)
  → [dilated block: (1, C, T) → (1, C, T) residual + (1, C, T) skip] × N
  → sum skips → (1, C, T)
  → ReLU → 1x1 → ReLU → 1x1 → (1, 256, T)
  → softmax → categorical distribution per timestep
```

## Deliberate Simplifications vs. Full Paper

1. **Toy audio data**: We use synthetic sine wave mixtures instead of real speech (e.g., VCTK corpus). The paper trained on 20+ hours of multi-speaker audio.
2. **Reduced scale**: 10 layers, 1 stack, 32 channels (paper used 30 layers, 3 stacks, 128-512 channels). Receptive field = 1024 vs paper's larger.
3. **No conditional generation**: We implement unconditional WaveNet only (paper includes speaker-conditioned and text-conditioned variants).
4. **Fewer epochs**: 20-30 epochs vs paper's extensive training.
5. **8 kHz sample rate**: Paper uses 16 kHz; we use 8 kHz to reduce sequence length.
6. **No context stacks**: The paper uses multi-resolution context stacks; we use a single stack.
7. **Simpler generation**: We use basic categorical sampling without temperature annealing.
