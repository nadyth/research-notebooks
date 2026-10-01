# Aggregated Residual Transformations for Deep Neural Networks (ResNeXt)

**arXiv:** [https://arxiv.org/abs/1611.05431](https://arxiv.org/abs/1611.05431)
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/aggregated-residual-transformations-resnext)

**Authors:** Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He (UC San Diego / Facebook AI Research)
**Published:** CVPR 2017

## Summary

ResNeXt introduces **cardinality** — the number of independent transformation paths in a building block — as a new dimension for scaling neural networks, alongside the traditional dimensions of depth (number of layers) and width (number of channels). The core idea is deceptively simple: instead of a single wide transformation, split the input into multiple lower-dimensional embeddings, apply the same topology of transformation to each, and aggregate the results by summation. All paths share the same architecture, so the only new hyperparameter is *C* (cardinality), the number of paths.

The paper shows that a ResNeXt-50 with 32 groups (denoted 32×4d) has roughly the same number of parameters and FLOPs as a ResNet-50, yet achieves higher accuracy on ImageNet-1K. Increasing cardinality is more effective at improving accuracy than increasing depth or width when the model capacity is scaled up. A 101-layer ResNeXt outperforms a 200-layer ResNet while using only 50% of the FLOPs. The key architectural insight is that the multi-branch aggregated transformation can be equivalently implemented using **grouped convolutions** — a technique dating back to AlexNet, originally used to split a model across GPUs. ResNeXt repurposes grouped convolutions as an accuracy-improving mechanism rather than an engineering compromise. The design follows two simple rules inherited from VGG/ResNet: (i) blocks producing the same spatial map size share hyperparameters, and (ii) width doubles when spatial resolution halves. This results in a homogeneous architecture with only a few hyperparameters to tune.

ResNeXt was the foundation of FAIR's ILSVRC 2016 classification submission, which secured 2nd place. The official code is at https://github.com/facebookresearch/ResNeXt.

## What problem does it solve

Imagine you run a factory that makes widgets. You have one big machine that does everything — cutting, shaping, and painting — all in one go. It works, but it's expensive and hard to improve. You could make the machine bigger (wider) or add more steps to the assembly line (deeper), but each upgrade gives smaller and smaller improvements.

The ResNeXt paper says: instead of one giant machine, use **32 small identical machines** working in parallel, each doing the same job on a smaller piece of material. Then combine all their outputs. Even though the total work is the same as the single big machine, the parallel setup produces better results. And here's the key finding: adding more small machines (increasing cardinality) gives you bigger improvements than making the big machine bigger or the assembly line longer.

Think of it like cooking. One master chef doing everything will be limited by how much they can handle. But 32 cooks each making the same dish in parallel, then combining the best of each — you get a better meal. The "cardinality" is just how many cooks you have. The paper proved that hiring more cooks is a better strategy than making the kitchen bigger or adding more steps to the recipe.

## Influence

ResNeXt is one of the most influential CNN architectures. The concept of cardinality and grouped convolutions for accuracy (not just efficiency) directly influenced EfficientNet's compound scaling, SE-Net's channel attention, and modern architectures that use multi-branch designs. ResNeXt-50 and ResNeXt-101 are standard backbones in torchvision and are widely used for detection (Mask R-CNN), segmentation, and transfer learning. The paper has been cited over 10,000 times and is a staple in computer vision courses. The split-transform-merge paradigm it formalized became a guiding principle for subsequent architecture design.
