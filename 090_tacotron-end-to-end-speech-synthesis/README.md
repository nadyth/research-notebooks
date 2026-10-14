# Tacotron: Towards End-to-End Speech Synthesis

**Paper:** [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135)
**Authors:** Yuxuan Wang, RJ Skerry-Ryan, Daisy Stanton, Yonghui Wu, Ron J. Weiss, Navdeep Jaitly, Zongheng Yang, Ying Xiao, Zhifeng Chen, Samy Bengio, Quoc Le, Yannis Agiomyrgiannakis, Rob Clark, Rif A. Saurous (Google, 2017)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/tacotron-towards-end-to-end-speech-synthesis)

## Summary

Tacotron is an end-to-end generative text-to-speech (TTS) model that synthesizes speech directly from characters — no phoneme-level alignment, no hand-engineered linguistic features, and no complex multi-stage pipeline. The architecture is a sequence-to-sequence model with attention: a **CBHG encoder** (1-D convolution bank + highway network + bidirectional GRU) extracts robust sequential representations from character embeddings; an attention-based decoder autoregressively predicts mel-spectrogram frames, outputting r=2 frames per decoder step for faster convergence and smoother alignment; and a **post-processing CBHG net** refines the predicted spectrogram using full-sequence bidirectional context before Griffin-Lim reconstruction converts it to a waveform. Key techniques include pre-nets with bottleneck dropout for generalization, the multi-frame reduction factor r to stabilize attention learning, and reconstructing zero-padded frames so the model learns when to stop. Tacotron achieves a 3.82 mean opinion score (MOS) on US English, outperforming a production parametric TTS system in naturalness.

## What problem does it solve

Imagine you type a sentence on your phone and want it read aloud in a natural-sounding human voice — like when your GPS says "turn left" but you wish it sounded more like a real person, not a robot. The old way of making computers talk was like a factory assembly line: one machine figures out how to pronounce each word, another machine predicts the sound shapes, and a third machine stitches the sounds together into audio. Each machine was built by hand with lots of tricky rules, and if any machine made a small mistake, the final voice sounded weird or robotic. Tacotron solves this by replacing the entire factory with a single AI that learns directly from examples — you give it thousands of pairs of "text + matching audio" and it figures out on its own how to turn new text into speech. No hand-built rules, no separate stages — one model that learns the whole process end-to-end, producing speech that human listeners rated as more natural than the old factory-style systems.

## Core Idea

- **End-to-end character-to-spectrogram**: The model takes raw characters as input and outputs mel-spectrogram frames, trained from scratch with random initialization on <text, audio> pairs.
- **CBHG module**: A reusable building block (Convolution Bank + Highway network + bidirectional GRU) used in both the encoder and post-processing net. The convolution bank with K filter widths (1 through K) models n-gram-like local context; max pooling adds local invariance; highway networks extract high-level features; bidirectional GRU captures sequential context.
- **Attention-based decoder**: Content-based tanh attention with a stateful GRU producing the attention query at each step. The decoder predicts r=2 mel-spectrogram frames per step (reduction factor), which divides decoder steps by r and dramatically speeds up alignment learning.
- **Pre-net with dropout bottleneck**: Both encoder and decoder inputs pass through a pre-net (FC-256-ReLU → Dropout(0.5) → FC-128-ReLU → Dropout(0.5)). The dropout acts as a noise source that helps the model generalize and handle the multi-modal nature of speech outputs.
- **Post-processing net**: A CBHG module that sees the entire decoded sequence (bidirectional context) and refines the mel-spectrogram into linear-frequency spectrogram with better harmonic resolution before Griffin-Lim synthesis.
- **Griffin-Lim waveform synthesis**: Converts the predicted linear spectrogram to audio using the iterative Griffin-Lim algorithm (50 iterations). Predicted magnitudes are raised to power 1.2 to reduce artifacts.
- **Loss**: Simple L1 loss on both the mel-spectrogram (seq2seq decoder) and linear spectrogram (post-processing net), with equal weights.

## Key Method Details

- **Spectral analysis**: 24 kHz sample rate, 50 ms frame length, 12.5 ms frame shift, 2048-point FFT, Hann window, pre-emphasis 0.97, 80-band mel-scale spectrogram.
- **Encoder**: Character one-hot → 256-D embedding → pre-net (with dropout) → CBHG (K=16 conv filters, 128 channels, highway net 4 layers, bidirectional GRU 128 cells).
- **Decoder**: Pre-net (same as encoder) → attention RNN (1-layer GRU, 256 cells) → 2-layer residual GRU decoder (256 cells) → FC output layer predicting 80 × r mel frames.
- **Post-processing**: CBHG (K=8, 128 channels) → projections to 80-dim linear spectrogram.
- **Training**: Adam optimizer, lr 0.001 → 0.0005 → 0.0003 → 0.0001 at 500K/1M/2M steps, batch size 32, L1 loss.
- **Reduction factor r=2**: Predicting multiple frames per decoder step stabilizes attention and reduces inference time.

## Influence

Tacotron established the end-to-end neural TTS paradigm and spawned a direct lineage of influential models: Tacotron 2 (2017, combined with WaveNet vocoder for near-human quality), the Google TTS production system, and the broader family of attention-based spectrogram prediction models (Deep Voice 3, Transformer TTS, FastSpeech). The CBHG module, attention-based alignment for speech, and the character-to-spectrogram framework became standard building blocks in TTS research. The paper has been cited thousands of times and was a foundational milestone in making neural TTS practical. Demo page: https://google.github.io/tacotron/

**arXiv:** https://arxiv.org/abs/1703.10135
