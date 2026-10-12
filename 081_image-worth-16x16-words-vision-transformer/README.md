# An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)

**arXiv:** https://arxiv.org/abs/2010.11929

**Authors:** Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Research, Brain Team)

**Published:** ICLR 2021 (submitted Oct 2020, revised Jun 2021)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/image-worth-16x16-words-vision-transformer)

## Summary

The Vision Transformer (ViT) paper demonstrates that a standard Transformer architecture — originally designed for NLP — can be applied directly to image classification tasks without relying on convolutional neural networks. The key insight is to split an image into fixed-size patches (e.g., 16×16 pixels), linearly embed each patch, add positional embeddings, and feed the resulting sequence into a standard Transformer encoder. A learnable [CLS] token prepended to the sequence serves as the image representation for classification. When pre-trained on large datasets (14M–300M images), ViT matches or exceeds state-of-the-art CNNs on ImageNet (88.55%), CIFAR-100 (94.55%), and VTAB while requiring substantially fewer computational resources to train. The paper argues that large-scale training trumps the inductive biases (locality, translation equivariance) baked into CNNs.

## Core Idea

The central methodological contribution is applying the vanilla Transformer encoder (Vaswani et al., 2017) to images with minimal modification. An image x ∈ R^(H×W×C) is reshaped into N = HW/P² patches of size P×P, each flattened and linearly projected to dimension D. A learnable [CLS] token is prepended, and learnable 1D positional embeddings are added. The resulting sequence of N+1 vectors is processed through L Transformer encoder layers, each consisting of multihead self-attention (MSA) and an MLP block with GELU activation, with LayerNorm before each block and residual connections after. The [CLS] token output serves as the classification representation.

## What Problem Does It Solve

Imagine you have a big box of photos and you want a computer to tell you what's in each one — a dog, a car, a flower. For many years, computers used a special technique called "CNN" to look at photos, which worked like sliding a tiny magnifying glass across the image bit by bit to find patterns. This worked well but was complicated and specific to images. Meanwhile, a different tool called a "Transformer" had become super popular for understanding language (like how your phone autocompletes sentences) — it looked at all the words at once and figured out which ones were most important. The problem was: Transformers worked great for words, but nobody could get them to work well for pictures without also using the old CNN magnifying-glass trick. This paper showed that you can just cut a picture into small squares (like a jigsaw puzzle with 16×16 pieces), treat each piece like a "word," and feed them into the same Transformer that's used for language. If you train it on enough pictures, it actually does BETTER than the old CNN method! It's like proving you don't need the special magnifying glass at all — the same tool that's great at reading sentences is also great at understanding photos, as long as you give it enough practice.

## Key Method Details

- **Patch embedding:** Image split into non-overlapping P×P patches, flattened, linearly projected to D dimensions
- **[CLS] token:** Learnable embedding prepended; its final-layer output is the image representation
- **Positional embedding:** Standard learnable 1D position embeddings (N+1 entries)
- **Transformer encoder:** L layers of {LayerNorm → MSA → residual; LayerNorm → MLP(GELU) → residual}
- **Model variants:** ViT-Base (L=12, D=768, heads=12, 86M params), ViT-Large (L=24, D=1024, heads=16, 307M), ViT-Huge (L=32, D=1280, heads=16, 632M)
- **Training:** Supervised classification on large datasets (ImageNet-21k, JFT-300M); fine-tune on downstream tasks
- **Key finding:** ViT underperforms CNNs on small datasets due to lack of inductive bias, but surpasses them when pre-trained on sufficient data

## Influence

ViT is one of the most influential computer vision papers of the decade. It sparked the "Vision Transformer" revolution, leading to variants like DeiT, Swin Transformer, BEiT, and MAE. It fundamentally shifted the CV community away from CNN-centric thinking toward Transformer-based architectures. The paper has been cited over 100,000 times and the ViT architecture forms the backbone of modern multimodal models like CLIP, DALL-E, and many others. The official code is at https://github.com/google-research/vision_transformer.
