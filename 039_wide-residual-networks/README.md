# Wide Residual Networks

**arXiv:** [https://arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146)
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/wide-residual-networks)

**Authors:** Sergey Zagoruyko, Nikos Komodakis (Université Paris-Est, École des Ponts ParisTech)

## Summary

Wide Residual Networks (WRNs) challenge the prevailing trend of making ResNets ever deeper and thinner. While ResNet demonstrated that hundreds or thousands of layers can be trained with skip connections, the authors observed that each marginal accuracy gain required roughly doubling the depth — and that very deep residual networks suffered from **diminishing feature reuse**, where many residual blocks contribute little to the final prediction because gradient can flow entirely through the identity shortcut, bypassing the block weights.

The core idea is simple: instead of stacking more layers, **widen** each residual block by increasing the number of feature channels by a factor *k* (the widening factor). A WRN-n-k denotes a network with *n* total layers and width multiplier *k*. The paper shows that a 16-layer-wide network (WRN-16-10) matches or beats a 1000-layer-thin network in accuracy while being several times faster to train. The authors also introduce **dropout between convolutional layers** inside residual blocks (not on the identity path), which provides consistent regularization gains as width increases the parameter count.

Key experimental findings: (1) width is more efficient than depth for improving ResNet performance; (2) a 16-layer WRN outperforms all prior deep ResNets on CIFAR-10, SVHN, COCO, and ImageNet; (3) dropout in residual blocks compensates for the increased overfitting risk of wider networks; (4) the main power of residual networks comes from the residual blocks themselves, with depth being supplementary. WRN-28-10 achieved 3.89% error on CIFAR-10 and 1.64% on SVHN — state-of-the-art at the time.

## What problem does it solve

Imagine you're building a tower out of Lego blocks. Each block is a layer in a neural network. Researchers found that making the tower taller (more layers) made the network better — but only barely. Each time you doubled the height, you got just a tiny improvement, and the tower became wobbly: some blocks in the middle weren't really doing anything useful anymore because the signal could just bypass them through a shortcut slide.

The Wide Residual Networks paper says: instead of building a super tall, skinny tower, build a **wider, shorter one**. Make each block fatter (more channels) rather than adding more blocks. It turns out a short, wide tower is just as strong as a skyscraper — and it's much faster to build. Think of it like hiring 10 skilled workers at each floor of a 5-story building, instead of having 1 worker on each floor of a 50-story building. Same total workforce, but the shorter building is easier to manage and gets the job done faster.

## Influence

Wide Residual Networks became one of the most cited ResNet variants. The insight that width matters as much as depth influenced subsequent architecture design (ResNeXt, EfficientNet). WRNs are widely used as baselines in transfer learning, semi-supervised learning, and neural architecture search. The official code (https://github.com/szagoruyko/wide-residual-networks) has been forked and adapted extensively. The paper has been cited thousands of times and remains a standard reference for efficient CNN architecture design.
