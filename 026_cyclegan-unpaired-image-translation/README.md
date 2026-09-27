# CycleGAN: Unpaired Image-to-Image Translation

**Paper:** [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
**Authors:** Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros
**Year:** 2017 (ICCV 2017)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/cyclegan-unpaired-image-translation)

## Summary

CycleGAN learns to translate images from one domain to another (e.g., horses → zebras, summer → winter, photos → paintings) **without paired training examples**. Previous image-to-image translation methods like Pix2Pix require aligned input-output pairs, which are expensive or impossible to collect for many tasks. CycleGAN sidesteps this by training two generator-discriminator pairs simultaneously: one mapping domain X → Y and another mapping Y → X, connected by a **cycle consistency loss** that ensures an image translated to the other domain and back reconstructs the original (F(G(x)) ≈ x). This removes the need for paired data while producing high-quality results.

## Core Idea

The method introduces two generators G (X→Y) and F (Y→X), and two discriminators D_Y (distinguishes real Y from fake G(x)) and D_X (distinguishes real X from fake F(y)). The total loss combines:

1. **Adversarial losses** — G tries to fool D_Y into thinking G(x) is real Y; F tries to fool D_X.
2. **Cycle consistency loss** — L1 distance between F(G(x)) and x, and between G(F(y)) and y. This is the key innovation: it regularizes the mapping so the generators can't just produce arbitrary images in the target domain.

The generators use a residual architecture with several residual blocks (9 for 256×256, fewer for smaller images). Discriminators use a 70×70 PatchGAN classifier that classifies local patches as real/fake.

## What problem does it solve

Imagine you have a bunch of photos of horses and a separate bunch of photos of zebras — but nobody has ever taken a photo of a horse standing in the exact same pose as a zebra so you could compare them side by side. You want a computer program that can turn any horse photo into a zebra photo. Older programs needed "before and after" pairs of the same image, like showing a horse and the exact same horse turned into a zebra. But that doesn't exist! CycleGAN solves this by learning from two separate piles — one of horses, one of zebras — with no matching pairs at all. It does this by making sure that if you turn a horse into a zebra and then back into a horse, you get the original horse again. This "round-trip" check teaches the computer to make realistic transformations without ever needing paired examples.

## Key Method Details

- **Generators:** ResNet architecture — 2-3 convolutional downsampling layers, 6-9 residual blocks, 2-3 transposed convolution upsampling layers. Instance normalization throughout.
- **Discriminators:** 70×70 PatchGAN — 4 convolutional layers classifying NxN patches.
- **Losses:** Adversarial (LSGAN / MSE form) + cycle consistency (L1, weight λ=10) + optional identity loss (L1, weight λ_identity=0.5).
- **Training:** Two Gs, two Ds trained alternately. Adam optimizer, lr=0.0002, linear decay after 100 epochs.
- **Simplifications in this notebook:** Smaller image resolution (128×128), fewer residual blocks (4-6), synthetic or small dataset, reduced epochs for Kaggle runtime.

## Influence

CycleGAN is one of the most influential papers in image-to-image translation, cited 25,000+ times. It spawned a line of research including UNIT, MUNIT, DRIT, and extensions to multi-domain translation (StarGAN). It remains widely used for artistic style transfer, domain adaptation, and data augmentation.
