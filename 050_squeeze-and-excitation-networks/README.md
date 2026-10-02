# Squeeze-and-Excitation Networks (SENet)

**arXiv:** [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
**Authors:** Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
**Published:** 2017 (CVPR 2018)
**Code:** [https://github.com/hujie-frank/SENet](https://github.com/hujie-frank/SENet)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/squeeze-and-excitation-networks)

## Summary

Squeeze-and-Excitation Networks (SENet) introduce a lightweight architectural unit — the **SE block** — that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels. Standard convolution operators fuse spatial and channel information within local receptive fields, but the channel relationships remain implicit and local. The SE block addresses this gap in two steps: (1) **Squeeze** — global average pooling collapses each channel's spatial dimensions into a single descriptor, capturing global context; (2) **Excitation** — a bottleneck FC network (two small layers with a reduction ratio *r*) generates per-channel gating weights via a sigmoid, which are then used to scale the original feature maps. SE blocks can be dropped into any existing architecture (ResNet, Inception, ResNeXt) at modest computational cost (~10% more parameters, negligible FLOPs) and consistently improve accuracy. SENet formed the foundation of the ILSVRC 2017 classification submission that won first place, reducing top-5 error to 2.251% — a ~25% relative improvement over the 2016 winner.

## Core Idea

The key insight is that not all channels in a feature map are equally important for a given input. An SE block learns to **dynamically emphasize informative channels and suppress less useful ones** on a per-image basis. The squeeze step (global average pooling) produces a channel-wise statistic vector **z ∈ ℝᶜ**. The excitation step passes z through two FC layers: `C → C/r → C` with ReLU then sigmoid, producing channel-wise scale factors **s ∈ ℝᶜ**. The original feature maps U are then channel-wise multiplied by s: `F_scale(u_c, s_c) = s_c · u_c`. The reduction ratio *r* (typically 16) controls the bottleneck. This mechanism is end-to-end learnable and adds negligible computational overhead.

## Key Method Details

- **Squeeze (Global Information Embedding):** Global average pooling over H×W spatial dimensions per channel: `z_c = (1/(H×W)) Σᵢ Σⱼ u_c(i,j)`
- **Excitation (Adaptive Recalibration):** Two FC layers with bottleneck: `s = σ(W₂ · δ(W₁ · z))` where W₁ ∈ ℝ^(C/r × C), W₂ ∈ ℝ^(C × C/r), δ = ReLU, σ = sigmoid
- **Scale:** Channel-wise multiplication: `x̃_c = s_c · u_c`
- **Reduction ratio r:** Controls capacity of excitation network. r=16 works best across architectures.
- **Integration:** SE blocks are placed after the residual transformation in ResNet modules, before the skip connection addition.
- **Computational cost:** SE-ResNet-50 adds ~2.5M parameters (~10% increase) and ~0.01 GFLOPs compared to ResNet-50's ~3.86 GFLOPs, yet matches ResNet-101 (~7.58 GFLOPs) accuracy.

## What Problem Does It Solve

Imagine you're looking at a photo of a dog in a park. Your brain doesn't pay equal attention to every detail — you focus on the dog's fur, eyes, and shape (the important "channels" of information) and kind of tune out the grass texture in the background. Before SE blocks, neural networks treated every channel of features equally — like trying to look at everything in a photo with the same level of attention, which wastes energy on unimportant stuff. SE blocks give the network a tiny "attention manager" that learns to say "this channel is important for this image, turn it up!" and "this channel isn't useful right now, turn it down." It's like having a smart volume knob for each feature channel — making the important features louder and the noisy ones quieter, so the network makes better decisions with barely any extra work.

## Influence

- Won **ILSVRC 2017** (ImageNet classification) with 2.251% top-5 error — ~25% relative improvement over 2016
- SE blocks became a standard component in modern architectures (MobileNetV3, EfficientNet, etc.)
- Concept of channel attention spawned follow-up works: CBAM (spatial + channel attention), ECA-Net (efficient channel attention), GCNet, etc.
- One of the most cited computer vision papers (40,000+ citations as of 2024)
- Authors extended the idea to "Gather-Excite" in NIPS 2018, generalizing the spatial aggregation
