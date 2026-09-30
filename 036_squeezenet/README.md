# SqueezeNet: AlexNet-level accuracy with 50× fewer parameters and <0.5 MB model size

**Paper:** [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
**Authors:** Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, Kurt Keutzer (UC Berkeley / Stanford)
**Year:** 2016

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/squeezenet)

## Summary

SqueezeNet demonstrates that a carefully designed CNN architecture can achieve AlexNet-level accuracy on ImageNet while using **50× fewer parameters** — and when combined with deep compression, the model shrinks to under **0.5 MB**, making it 510× smaller than AlexNet. The key innovation is the **Fire module**, which "squeezes" information through a bottleneck of 1×1 convolutions before "expanding" with a mix of 1×1 and 3×3 convolutions. By stacking these Fire modules with strategic delayed downsampling and bypass connections, SqueezeNet matches AlexNet's accuracy with only 1.2 million parameters.

## What problem does it solve?

Imagine you have a big, heavy recipe book (a neural network) that works great but is so thick and heavy that it's hard to carry around. If you want to put it on a small shelf (like a phone or a tiny computer in a car), it doesn't fit. SqueezeNet is like rewriting that recipe book in a much shorter way — using fewer words but keeping all the important instructions so the recipes still come out just as good. This means you can fit the whole thing on a phone, a drone, or a self-driving car without needing a super-powerful computer. The researchers showed you don't need a giant model to get great results — you just need to be smart about how you design it.

## Core Idea

The **Fire module** has three hyperparameters:
- **s₁ₓ₁** (squeeze): number of 1×1 filters in the squeeze layer
- **e₁ₓ₁** (expand): number of 1×1 filters in the expand layer  
- **e₃ₓ₃** (expand): number of 3×3 filters in the expand layer

The squeeze layer reduces the number of input channels (bottleneck), then the expand layer increases the channel count using parallel 1×1 and 3×3 convolutions whose outputs are concatenated. The design principle is **s₁ₓ₁ < e₁ₓ₁ + e₃ₓ₃** — the squeeze acts as a bottleneck.

### Three architectural strategies:
1. **Replace 3×3 filters with 1×1 filters** where possible (9× fewer parameters per filter)
2. **Decrease the number of input channels** to 3×3 filters via squeeze layers
3. **Downsample late in the network** so feature maps retain maximum spatial resolution

### SqueezeNet architecture:
- **conv1**: 7×7, 96 filters, stride 2 → maxpool (3×3, stride 2)
- **fire2–fire5**: Fire modules (progressively wider) → maxpool after fire4
- **fire6–fire9**: Fire modules with bypass connections → maxpool after fire8
- **conv10**: 1×1, 1000 filters (classifier) → global avg pool

## Key Method Details

- **Fire module**: squeeze → expand (1×1 + 3×3 concatenated). The squeeze ratio is typically s₁ₓ₁ = (e₁ₓ₁ + e₃ₓ₃) / 3.
- **Bypass connections**: identity shortcuts around Fire modules (similar to ResNet) that improve accuracy by ~2.9%.
- **Delayed downsampling**: max-pooling placed later in the network (after fire4 and fire8) instead of early, which keeps larger feature maps for more of the forward pass and improves accuracy.
- **No fully-connected layers**: replaced with 1×1 convolution + global average pooling, drastically reducing parameters.

## Influence

SqueezeNet became one of the most cited lightweight architecture papers, influencing MobileNet, ShuffleNet, and the broader trend of efficient model design. It demonstrated that architectural innovation alone (without compression) could achieve dramatic parameter reduction, and that combining architecture design with model compression (Deep Compression) yields multiplicative savings. The Fire module pattern of squeeze-then-expand became a template for efficient convolutions in edge deployment scenarios.
