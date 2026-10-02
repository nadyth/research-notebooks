# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

**Paper:** [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
**Authors:** Mingxing Tan, Quoc V. Le (Google Research, Google Brain)
**Published:** ICML 2019 (arXiv: 1905.11946, May 2019)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/efficientnet-rethinking-model-scaling)

## Summary

EfficientNet is a landmark paper that systematically studies how to scale up convolutional neural networks (ConvNets). Instead of the common practice of scaling only one dimension (depth, width, or resolution), the authors show that **balancing all three dimensions** through a **compound scaling coefficient (φ)** yields dramatically better accuracy and efficiency. Starting from a small baseline network found via neural architecture search (EfficientNet-B0), the compound method scales depth as α^φ, width as β^φ, and resolution as γ^φ, where α·β²·γ² ≈ 2. The resulting family (B0–B7) achieves state-of-the-art accuracy with far fewer parameters and FLOPs — EfficientNet-B7 reaches 84.3% top-1 on ImageNet while being 8.4× smaller and 6.1× faster than the best previous ConvNet.

## Core Idea

The compound scaling method is based on an empirical observation: scaling up network width, depth, and resolution together is more effective than scaling any single dimension. The paper formalizes this as:

- **Depth (d):** scales as d = α^φ
- **Width (w):** scales as w = β^φ
- **Resolution (r):** scales as r = γ^φ
- **Constraint:** α · β² · γ² ≈ 2 (each step of φ roughly doubles FLOPs)

Where α, β, γ are determined by a small grid search on the baseline model, and φ is a user-specified coefficient controlling the overall scaling.

The baseline architecture (EfficientNet-B0) uses inverted bottlenecks from MobileNetV2 and squeeze-and-excitation optimization from SENet, found via neural architecture search.

## Key Method Details

1. **MBConv blocks:** Mobile inverted bottleneck convolution (from MobileNetV2) with squeeze-and-excitation (SE) optimization — each block has an expansion phase, depthwise conv, SE channel-attention, and a projection phase.
2. **Compound scaling:** α (depth), β (width), γ (resolution) found via grid search on B0; constraint α·β²·γ² ≈ 2.
3. **EfficientNet-B0:** Baseline model with ~5.3M parameters, ~77.1% ImageNet top-1 accuracy.
4. **Scaling family B0→B7:** φ ranges from 0 to 7; B7 reaches 84.3% top-1 with 66M parameters.

## What Problem Does It Solve

Imagine you're building a bookshelf. You could make it taller (more shelves = more depth), wider (more books per shelf = more width), or use bigger books (higher resolution = more detail per book). Most people just keep making the bookshelf taller — adding more and more shelves — but eventually the shelves get wobbly and the whole thing tips over.

This paper figured out something smart: if you make the bookshelf taller **and** wider **and** use bigger books **all at the same time and in the right balance**, you get a much sturdier, more useful bookshelf than if you only changed one thing. The "right balance" is the secret sauce — a simple formula that tells you exactly how much to grow each dimension so they work together perfectly.

Before this paper, AI researchers would just make their models deeper (more layers) to make them smarter, but that used a ton of computer power for tiny improvements. EfficientNet showed that by scaling everything together, you can get way smarter while using way less power — like getting a bookshelf that holds 8× more books but takes up less space.

## Influence

EfficientNet became one of the most widely adopted CNN architectures, with 40,000+ citations. It influenced:
- EfficientNetV2 (2021) with progressive learning and faster training
- Widespread use in transfer learning, medical imaging, and edge deployment
- Popularized compound scaling as a general principle for model scaling
- The baseline for many subsequent scaling studies and NAS papers
