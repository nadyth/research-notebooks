# Representation Learning with Contrastive Predictive Coding (CPC)

**Paper:** van den Oord, A., Li, Y., & Vinyals, O. (2018). *Representation Learning with Contrastive Predictive Coding.* arXiv:1807.03748  
**arXiv:** https://arxiv.org/abs/1807.03748  
**Citations:** 6,300+ (as of 2024, per OpenAlex)  

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/069-contrastive-predictive-coding)

---

## Summary

Contrastive Predictive Coding (CPC) is a universal unsupervised representation-learning framework that extracts useful features from high-dimensional sequential data — audio, images, text, and reinforcement-learning observations — without any labels. The core idea has two parts: (1) encode each observation into a compact latent vector with a non-linear encoder, then (2) use an autoregressive model to summarize past latents into a context vector that is trained to *predict future latents*. Instead of predicting raw future pixels or audio samples directly (which is computationally expensive and wastes capacity on low-level noise), CPC predicts in latent space and uses a **contrastive loss called InfoNCE**: given one true future sample and N−1 negative (random) samples, the model must pick the correct one. This loss is equivalent to a categorical cross-entropy over N candidates and, when optimized, maximizes a lower bound on the mutual information between the context and the future observation. The authors demonstrated strong results across speech (phone classification on LibriSpeech), vision (ImageNet classification), NLP (sentence classification on BookCorpus), and RL (DeepMind Lab), showing CPC is a domain-agnostic representation learner.

## Core Idea

1. **Encoder** g_enc: maps each input x_t → latent z_t (e.g., a 1-D conv for audio, ResNet for images, conv+pool for sentences).
2. **Autoregressive model** g_ar (GRU or PixelCNN-style): summarizes z_{≤t} into a context vector c_t.
3. **Density ratio** f_k(x_{t+k}, c_t) = exp(z_{t+k}^T W_k c_t): a log-bilinear model scores how well a candidate future latent matches the context prediction at horizon k.
4. **InfoNCE loss**: treats the problem as N-way classification — one positive future sample vs N−1 negatives drawn from the marginal. L_N = -E[log( f_k(x_pos, c_t) / Σ f_k(x_i, c_t) )].
5. Minimizing InfoNCE maximizes a lower bound on I(x_{t+k}; c_t) ≥ log(N) − L_N^opt.

## Key Method Details Relevant to the Code

- **InfoNCE loss** is the central implementation target: it is computed per prediction step k, over a batch where each sample's true future is the positive and all other samples' futures serve as negatives.
- **Log-bilinear scoring**: f_k = exp(z_{t+k} · W_k · c_t), where W_k is a learned linear projection per step k.
- **Multi-step prediction**: predicting multiple future steps simultaneously (not just t+1) produces richer representations.
- **Negative sampling strategy**: the paper compares "sample" (within-batch), "same-sequence" (from other time-steps of same sequence), and "mixed" negatives.
- **Downstream evaluation**: a linear classifier is trained on frozen CPC representations to measure linear separability.

## What problem does it solve

Imagine you're trying to learn a language by only listening to people talk — no dictionary, no translations, nobody telling you what words mean. How would you figure out what's going on? CPC solves a similar problem for AI: it helps computers learn useful features from raw data (like audio, images, or text) without anyone labeling anything.

Here's the trick: instead of trying to predict the exact next sound or pixel (which is really hard and wasteful), CPC compresses everything into a simpler "summary code" and then plays a guessing game. It looks at what happened so far, makes a prediction about what comes next in this simplified code space, and then checks: "Out of these N options, which one is the REAL thing that actually came next?" It's like a multiple-choice quiz where the model has to spot the correct future among decoys. By getting good at this quiz, the model automatically learns representations that capture the important, slow-changing patterns in the data — patterns that turn out to be super useful for downstream tasks like recognizing speech, classifying images, or understanding sentences.

## Influence

CPC introduced the **InfoNCE loss**, which became one of the most influential losses in modern self-supervised learning. It directly inspired SimCLR (2020), MoCo (2020), CLIP (2021), and many other contrastive methods that now dominate self-supervised representation learning. The idea of "predict future in latent space, contrast against negatives" became a foundational paradigm. The paper has 6,300+ citations and is considered a landmark in unsupervised representation learning.
