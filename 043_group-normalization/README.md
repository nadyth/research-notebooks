# Group Normalization

**Paper:** Wu & He (2018). *Group Normalization.* arXiv:1803.08494
**Link:** https://arxiv.org/abs/1803.08494

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/group-normalization)

## Summary

Group Normalization (GN) is a normalization technique that divides the channels of a feature map into groups and computes within each group the mean and variance for normalization. Unlike Batch Normalization (BN), which normalizes along the batch dimension and thus depends on batch statistics, GN's computation is independent of batch sizes. This makes GN particularly effective when batch sizes are small — a common constraint in object detection, segmentation, and video classification tasks where memory limits batch size. On ResNet-50 trained on ImageNet, GN has 10.6% lower error than its BN counterpart when using a batch size of 2, and at typical batch sizes GN is comparably good with BN and outperforms other normalization variants like LayerNorm and InstanceNorm. GN is also naturally transferable from pre-training to fine-tuning, and outperforms BN-based counterparts for object detection/segmentation in COCO and video classification in Kinetics.

**Core idea:** Instead of normalizing across the batch dimension (BatchNorm), GN partitions the C channels into G groups, and normalizes each group independently by computing mean and variance over the (C/G × H × W) pixels within each group for each sample. This is equivalent to Layer Normalization when G=1 and Instance Normalization when G=C.

**Key method details:**
- For a feature map of shape (N, C, H, W), GN divides C channels into G groups, each with C/G channels
- For each sample n and each group g, compute μ and σ² over the C/G × H × W elements
- Normalize: x̂ = (x - μ) / sqrt(σ² + ε), then apply learnable scale γ and shift β per channel
- Computation is entirely within each sample — no dependency on other samples in the batch
- Same computation at training and inference time (no running averages needed)
- GN can be implemented in a few lines of code in PyTorch/Facebook's detectron

**Influence:** Group Normalization has been widely adopted in computer vision, particularly in object detection and segmentation pipelines where small batch sizes are standard. It is built into PyTorch as `nn.GroupNorm` and is the default normalization in many detection frameworks (e.g., Detectron2's mask R-CNN variants). The paper has been cited thousands of times and GN remains a standard alternative to BatchNorm for memory-constrained training.

## What problem does it solve?

Imagine you're a teacher grading tests. To be fair, you compare each student's score to the average score of the whole class — that's what Batch Normalization does: it compares each example to the average of a "batch" of examples.

But what if your class is very small? Maybe there are only 2 students. The "average" of 2 students is not very reliable — if one happens to be a genius and the other is having a bad day, the average is skewed and the grading becomes unfair. This is exactly what happens in deep learning: when you can only fit 2 images in GPU memory at a time (common for large images like in object detection), Batch Normalization's statistics become unreliable and the model's accuracy drops a lot.

Group Normalization solves this by not looking at other students at all. Instead, it divides each student's test into sections (groups of channels) and normalizes each section on its own. This means:
- You can grade a single test accurately (doesn't need a big batch)
- The grading is the same whether you're practicing or in the real exam (same at train and test time)
- It works great for big, complex tasks like detecting objects in high-resolution images

This simple idea made it possible to train high-quality models for detection and segmentation even when memory limits you to tiny batch sizes.

## Kaggle

[Open in Kaggle](https://www.kaggle.com/code/nadymsazad/group-normalization)
