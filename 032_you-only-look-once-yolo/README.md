# You Only Look Once: Unified, Real-Time Object Detection (YOLO)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/code/nadymsazad/you-only-look-once-yolo)

**arXiv:** [1506.02640](https://arxiv.org/abs/1506.02640) | **Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi (2015)

## Summary

YOLO reframes object detection as a single regression problem. Instead of running a classifier over thousands of candidate regions (as R-CNN and DPM do), YOLO uses a single convolutional network to simultaneously predict all bounding boxes and class probabilities for an entire image in one forward pass. The image is divided into a 7×7 grid; each grid cell predicts bounding box coordinates (x, y, w, h) and class-conditional probabilities for the objects whose centers fall within that cell. This unified architecture processes images at 45 FPS (155 FPS for Fast YOLO), making it the first real-time deep-learning detector. The network is trained end-to-end with a multi-part sum-squared-error loss that balances localization and classification errors, using λ = 4 to up-weight coordinate predictions. Bounding box dimensions use square-root parameterization so that errors in small boxes are penalized proportionally more than errors in large boxes.

## What problem does it solve

Imagine you're looking at a photo of a busy street and you need to quickly point out every person, car, and bicycle — not just say "there's a car somewhere" but draw a box around each one and label it. Before YOLO, computers did this in a very roundabout way: they'd cut the photo into thousands of tiny pieces, look at each piece one by one through a "what is this?" classifier, and then try to stitch the answers back together. This was slow (taking seconds per photo) and often resulted in the computer "seeing" objects that weren't really there because it was looking at tiny patches without understanding the whole picture. YOLO solved this by looking at the entire photo all at once — just like a human does — and directly predicting where every object is and what it is, in a single glance. That's why it's called "You Only Look Once."

## Core Idea

- **Unified detection:** A single CNN predicts bounding boxes and class probabilities simultaneously from the full image — no region proposal step, no separate classifier.
- **Grid-based prediction:** The input image (448×448) is divided into a 7×7 grid. Each grid cell predicts B bounding boxes (the paper uses B=2 in the full model, B=1 in the simplified version described here) with confidence scores, plus C conditional class probabilities (20 classes for PASCAL VOC).
- **Output tensor:** The final layer produces a 7×7×(5*B + C) tensor. With B=2 and C=20, that's 7×7×30.
- **Loss function:** Multi-part sum-squared error with λ_coord = 4 (up-weighting localization) and λ_noobj = 0.5 (down-weighting boxes without objects). Square-root parameterization for width/height.
- **Architecture:** 24 convolutional layers (inspired by GoogLeNet, using 1×1 reduction + 3×3 conv layers) followed by 2 fully connected layers. Strided convolutions replace maxpooling for downsampling.
- **Training:** Pretrained on ImageNet (first 20 conv layers, ~1 week), then fine-tuned for detection on PASCAL VOC 2007/2012. ~120 epochs, batch size 64, momentum 0.9, dropout 0.5.
- **Speed:** 45 FPS (base), 155 FPS (Fast YOLO with 9 conv layers).

## Key Method Details

- **Objectness score:** Each predicted bounding box carries a confidence = Pr(Object) × IOU(pred, truth). This decouples "is there something?" from "what is it?" and handles the class imbalance from mostly-empty grid cells.
- **Conditional class probabilities:** Pr(Class_i | Object) — only updated at cells containing an object. Final detection score = Pr(Class_i | Object) × Pr(Object) × IOU.
- **Non-maximal suppression:** Post-processing to remove duplicate detections of the same object; adds 2-3% mAP.
- **Leaky ReLU** activation (0.1 slope for negative inputs) throughout, with sigmoid on the final layer to bound outputs to [0, 1].

## Simplifications in This Notebook

- Smaller network (fewer conv layers) for Kaggle GPU runtime constraints
- Synthetic toy dataset (colored shapes on plain backgrounds) instead of PASCAL VOC
- B=1 bounding box per cell (original paper uses B=2)
- Fewer classes (3-5 instead of 20)
- Shorter training (~20-30 epochs instead of 120)
- No ImageNet pretraining; trained from scratch on the synthetic dataset

## Influence

YOLO revolutionized object detection by demonstrating that a single neural network could achieve real-time detection speed. It spawned an entire family of models (YOLOv2/YOLO9000, YOLOv3, YOLOv4, YOLOv5, YOLOv8, YOLOv10, YOLO11) that remain among the most widely used detectors in both research and industry. The original YOLO paper has been cited 40,000+ times and is one of the most influential computer vision papers of the 2010s. Its framing of detection as a regression problem influenced single-stage detectors like SSD and RetinaNet.
