# Zero-Shot Text-to-Image Generation (DALL-E)

**arXiv:** https://arxiv.org/abs/2102.12092

**Authors:** Aditya Ramesh, Mikhail Pavlov, Gabriel Goh, Scott Gray, Chelsea Voss, Alec Radford, Mark Chen, Ilya Sutskever (OpenAI)

**Year:** 2021

## Summary

DALL-E introduces a simple yet powerful approach to text-to-image generation: a single autoregressive Transformer that models both text and image tokens as one continuous stream. The method has two stages. First, a discrete Variational Autoencoder (dVAE) compresses each 256×256 image into a 32×32 grid of image tokens from a learned codebook of 8192 entries — turning images into a sequence of discrete symbols. Second, a 12-billion-parameter decoder-only Transformer (similar to GPT-3) is trained autoregressively on the concatenation of BPE-tokenized text captions (up to 256 tokens) and the dVAE image tokens (1024 tokens). At inference time, the model receives a text prompt, generates image tokens one by one, and the dVAE decoder converts those tokens back into pixels. A separately trained CLIP model reranks generated candidates to select the best match. Trained on 250 million text-image pairs, DALL-E demonstrates zero-shot generation: it can produce images for text descriptions never seen during training, exhibiting compositional reasoning like combining objects in novel ways, applying styles, and performing image-to-image edits.

## Core Idea

The key insight is that text-to-image generation can be cast as a **single sequence modeling problem**. Instead of designing specialized architectures with auxiliary losses, DALL-E treats image generation as language modeling — the same paradigm that powered GPT-3, but extended to a joint vocabulary of text and image tokens.

**Discrete VAE (dVAE):** Images cannot be fed directly into a Transformer, so DALL-E first trains a dVAE that encodes a 256×256 RGB image into a 32×32 grid of discrete tokens (codebook size 8192). The dVAE uses an encoder-decoder architecture with a vector quantization bottleneck (similar to VQ-VAE), trained with a reconstruction loss combined with a commitment loss and straight-through gradient estimation. This reduces the image to a sequence of 1024 tokens — small enough for a Transformer to model autoregressively.

**Autoregressive Transformer:** The Transformer receives `[BOS] + text_tokens + [SEP] + image_tokens + [EOS]` and is trained to predict the next token. Text tokens come from BPE (vocabulary 16384), and image tokens from the dVAE codebook (vocabulary 8192). The combined vocabulary is 16384 + 8192 = 24576 tokens. The model uses a decoder-only architecture with attention masks ensuring each position only attends to previous positions.

**Sampling:** Given a text prompt, the Transformer generates image tokens autoregressively (using top-k or nucleus sampling). The dVAE decoder then converts the 1024 image tokens back into a 256×256 image. Multiple candidates are generated and reranked by CLIP to pick the best.

## Key Method Details Relevant to the Code

1. **dVAE with vector quantization** — encoder maps image patches to a continuous latent, which is quantized to the nearest codebook entry via nearest-neighbor lookup; gradients flow through via straight-through estimator
2. **Joint vocabulary** — text and image tokens share a single vocabulary space, with image tokens offset by the text vocabulary size
3. **Autoregressive training** — standard next-token prediction (cross-entropy) over the concatenated text+image token sequence
4. **Causal attention** — the Transformer uses a causal mask so image tokens can attend to all preceding text and image tokens, but not future ones
5. **Two-stage training** — dVAE is trained first (to establish the token space), then the Transformer is trained on frozen dVAE tokens
6. **CLIP reranking** — at inference, multiple samples are generated and CLIP selects the best text-image match

## What Problem Does It Solve

Imagine you tell your friend: "Draw me a picture of a snail wearing a tiny hat made of cheese." Your friend can imagine it and draw it because they understand what snails, hats, and cheese look like, and they can *combine* those ideas. But before DALL-E, computers couldn't do this — they could only draw things they were specifically trained to draw, like cats or dogs, with lots of labeled examples.

DALL-E taught computers to do what your friend does: **understand words and turn them into pictures**, even for weird combinations the computer had never seen before. It learned by looking at millions of pictures with captions, figuring out that the word "snail" looks like a certain shape, "hat" looks like another shape, and "cheese" has a certain color. Then it could mix and match those shapes to draw something brand new — like a snail with a cheese hat — just from the words you typed. This was a huge deal because it meant you didn't need to show the computer every possible picture; you just needed to describe it in words.

## Influence

DALL-E was a landmark paper that kicked off the text-to-image generation revolution:
- **Direct successor — DALL-E 2 (2022):** Replaced the autoregressive Transformer with a diffusion model conditioned on CLIP embeddings, achieving photorealistic results
- **Stable Diffusion (2022):** Adopted the latent space concept (compressing images before generation) that DALL-E pioneered with the dVAE
- **Midjourney, Imagen (Google), Parti (Google):** All followed in DALL-E's footsteps, exploring different generation backbones but retaining the text-conditioned generation paradigm
- **Compositional reasoning benchmark:** DALL-E's ability to combine concepts in novel ways (e.g., "an armchair in the shape of an avocado") became a standard test for generative models
- **Mainstream AI art movement:** DALL-E's release in January 2021 (alongside CLIP) is widely considered the beginning of the AI-generated art era
- **Architecture influence:** The idea of tokenizing images and modeling them autoregressively influenced subsequent work like Parti, CM3leon, and unified multimodal models

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/083_zero-shot-text-to-image-generation-dall-e)
