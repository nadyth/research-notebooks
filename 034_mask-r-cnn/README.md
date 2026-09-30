# Mask R-CNN

**Paper:** [Mask R-CNN](https://arxiv.org/abs/1703.06870)
**Authors:** Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick (Facebook AI Research)
**Published:** 20 Mar 2017 (v3: 24 Jan 2018)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/mask-r-cnn)

## Summary

Mask R-CNN is a conceptually simple, flexible, and general framework for **object instance segmentation** — the task of detecting every object in an image and simultaneously generating a pixel-accurate segmentation mask for each individual instance. It extends **Faster R-CNN** by adding a third branch (alongside the existing class-label and bounding-box regression branches) that predicts a binary segmentation mask for each Region of Interest (RoI). The mask branch is a small Fully Convolutional Network (FCN) applied to each RoI in a pixel-to-pixel manner. The key innovation beyond simply adding a mask head is **RoIAlign**, a quantization-free feature extraction layer that uses bilinear interpolation to preserve exact spatial alignment between the feature map and the RoI — fixing a critical misalignment problem in RoIPool that devastated pixel-accurate mask quality. A second key design decision is **decoupling mask and class prediction**: instead of a per-pixel softmax over all classes (which couples segmentation with classification), Mask R-CNN predicts an independent binary mask (sigmoid) per class and relies on the separate classification branch to pick the correct label, removing inter-class competition. Mask R-CNN achieved state-of-the-art results on all three COCO challenge tracks (instance segmentation, bounding-box detection, and person keypoint estimation), surpassing the 2016 COCO challenge winners with a single model running at 5 fps. The code was released via Facebook Research's Detectron.

### What Problem Does It Solve

Imagine you're looking at a photo of a busy street with five cars parked side by side. A regular object detector can draw a box around each car and say "that's a car, that's a car, that's a car." But what if you want to know exactly which pixels belong to *this* specific car versus the one next to it? Drawing boxes isn't enough — you need to paint a precise outline around every single car so they don't blur together. This is called **instance segmentation**, and it used to be really hard because old methods either drew boxes (too coarse) or painted all cars as one big blob (couldn't tell instances apart). Mask R-CNN solves this by adding a "coloring" step on top of the box-drawing step: after finding each car's box, it also draws a pixel-perfect outline for that specific car. It's like having a smart assistant who not only spots every car but also carefully traces the exact shape of each one with a digital pen.

## Core Idea

Mask R-CNN's framework has three parallel outputs for each candidate object:

1. **Class label** (what category is this object?)
2. **Bounding box regression** (where exactly is the box?)
3. **Binary mask** (what pixels belong to this object?)

The mask branch is a small FCN that operates on a fixed-size RoI feature (e.g., 14×14), producing a K×m×m output tensor where K is the number of classes and m×m is the mask resolution. A per-pixel sigmoid (not softmax) is applied, and only the mask corresponding to the predicted class is used at inference.

### Key Method Details

- **RoIAlign:** Replaces RoIPool with quantization-free bilinear interpolation. RoIPool rounds coordinates to integer grid positions, introducing misalignment that destroys pixel-level accuracy. RoIAlign samples 4 points per bin using bilinear interpolation from nearby feature-map grid points, preserving exact spatial locations. This single change improves mask AP by ~3 points and AP75 by ~5 points.

- **Decoupled Mask & Class Prediction:** Each class gets an independent binary mask (sigmoid + binary cross-entropy loss). The classification branch handles category prediction; the mask branch only handles spatial layout. This avoids the inter-class competition that softmax-based FCN segmentation suffers from. Ablation shows sigmoid masks improve AP by +5.5 over softmax.

- **Multi-task Loss:** L = L_cls + L_box + L_mask. The mask loss L_mask is defined only on positive RoIs (IoU ≥ 0.5 with ground truth), averaging binary cross-entropy across the m×m mask for the ground-truth class k.

- **Backbone Architectures:** Evaluated with ResNet-50-C4, ResNet-101-C4, ResNet-50-FPN, ResNet-101-FPN, and ResNeXt-101-FPN. FPN (Feature Pyramid Network) backbones give the best accuracy/speed tradeoff.

- **Training:** 160k iterations, learning rate 0.02 (×0.1 at 120k), momentum 0.9, weight decay 0.0001, 8 GPUs, 2 images/GPU. Each image samples N RoIs (64 for C4, 512 for FPN) with 1:3 positive:negative ratio.

- **Inference:** 1000 proposals (FPN) → NMS → top-100 detection boxes → mask branch applied only to these 100.

### Results (COCO test-dev)

| Backbone | mask AP | AP50 | AP75 |
|---|---|---|---|
| ResNet-101-C4 | 33.1 | 54.9 | 34.8 |
| ResNet-101-FPN | 35.7 | 58.0 | 37.8 |
| ResNeXt-101-FPN | 37.1 | 60.0 | 39.4 |

Mask R-CNN ran at ~5 fps (200ms/frame) and surpassed all previous single-model entries on COCO instance segmentation, detection, and keypoint estimation.

## Influence

Mask R-CNN became the dominant baseline for instance segmentation and a foundational architecture for downstream computer vision research. The Detectron code release (later replaced by Detectron2) made it the standard reference implementation. Key impacts:
- Set the COCO instance segmentation benchmark standard for years
- RoIAlign became a standard layer in detection frameworks
- Extended to human pose estimation (keypoint detection) by treating each keypoint as a one-hot binary mask
- Influenced YOLACT, SOLOv2, and other real-time instance segmentation methods
- The "add a parallel branch" design philosophy influenced multi-task learning in detection

## arXiv Link

https://arxiv.org/abs/1703.06870
