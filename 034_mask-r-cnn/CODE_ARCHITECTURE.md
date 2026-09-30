# Code Architecture — Mask R-CNN Notebook

## Overview

This notebook implements a **simplified but faithful** Mask R-CNN pipeline for instance segmentation. It demonstrates the three key innovations of the paper: (1) the parallel mask prediction branch, (2) RoIAlign for pixel-accurate feature extraction, and (3) decoupled per-class binary masks (sigmoid, not softmax). We train on a **synthetic shape dataset** (colored rectangles, circles, triangles on a canvas) so the entire pipeline runs on a single GPU in minutes without requiring COCO downloads.

## Section-by-Section Breakdown

### 1. Setup & Imports
Installs `torch`, `torchvision`, and supporting libraries. All imports in one cell.

### 2. Synthetic Shape Dataset (`ShapeDataset`)
- Generates 128×128 images containing 1–4 random shapes (rectangle, circle, triangle).
- Each shape has a random color, position, and size.
- Returns: image tensor (3, 128, 128), and targets dict with:
  - `boxes`: (N, 4) bounding boxes in [x1, y1, x2, y2] format
  - `labels`: (N,) class indices (1=rect, 2=circle, 3=triangle, 0=background)
  - `masks`: (N, 128, 128) binary masks per instance
- Train/test split: 400 train / 100 test images.

### 3. Visualization Helpers
- `visualize_sample(idx)`: overlays boxes, labels, and colored masks on an image.
- Used to show dataset samples before training.

### 4. RoIAlign Implementation
- Custom `roi_align(features, rois, output_size, spatial_scale)` function.
- Takes feature maps (B, C, H, W) and RoIs (N, 5) where each RoI is [batch_idx, x1, y1, x2, y2] in image coordinates.
- Converts to feature-map coordinates using `spatial_scale`.
- For each RoI bin, samples 4 points via bilinear interpolation from the feature map.
- No quantization — the key innovation from the paper.
- Falls back to torchvision's `roi_align` if available (for speed) but the custom implementation is shown for clarity.

### 5. Backbone Network (`SimpleBackbone`)
- A small CNN producing feature maps at different scales.
- Architecture: 4 conv blocks (Conv2d → BatchNorm → ReLU → MaxPool2d), channels [16, 32, 64, 128].
- Output stride: 16 (128 / 8 ≈ 16 per pooling). Produces (B, 128, 8, 8) feature map from 128×128 input.
- Deliberately simpler than ResNet-50/101 — enough for the synthetic task.

### 6. RPN — Region Proposal Network
- Lightweight network on top of backbone features.
- 3×3 conv → two 1×1 conv heads: objectness (2 scores per anchor) and bbox deltas (4 per anchor).
- 3 anchor scales (32², 64², 96²) × 3 aspect ratios (0.5, 1.0, 2.0) = 9 anchors per location.
- `generate_anchors()`: creates anchor boxes at each feature-map location.
- During training: assigns positive/negative labels via IoU threshold (0.7/0.3), samples 256 anchors.
- Loss: RPN classification (binary CE) + RPN regression (smooth L1).

### 7. Mask R-CNN Head
- **Box head:** RoIAlign features (7×7) → fc(1024) → fc(1024) → class logits (K+1) + bbox deltas (4×(K+1)).
- **Mask head:** RoIAlign features (14×14) → conv 3×3(256) → conv 3×3(256) → conv 3×3(256) → deconv 2×2 stride 2(256) → conv 1×1(K+1).
- Output: (N, K+1, 28, 28) mask logits → sigmoid per pixel.
- Key: **independent binary masks per class** (sigmoid), not softmax.

### 8. Full Model (`MaskRCNN`)
- Combines backbone + RPN + head.
- `forward(images, targets)`: returns losses dict during training, predictions dict during inference.
- `inference(images)`: proposals → NMS → top-100 → mask prediction → threshold at 0.5.

### 9. Loss Functions
- **RPN loss:** cls (BCE) + box (smooth L1), same as Faster R-CNN.
- **Box loss:** cls (cross-entropy) + box (smooth L1), same as Fast R-CNN. Only on positive RoIs.
- **Mask loss:** binary cross-entropy per pixel, averaged over m×m, only on the ground-truth class mask. Defined only on positive RoIs (IoU ≥ 0.5).
- **Total:** L = L_rpn_cls + L_rpn_box + L_cls + L_box + L_mask

### 10. Training Loop
- 10 epochs, batch_size=2, lr=1e-3, Adam optimizer.
- Tracks total loss, box loss, mask loss separately.
- Synthetic dataset trains in ~3-5 minutes on a single GPU.

### 11. Visualization & Evaluation
- `visualize_prediction(idx)`: shows predicted boxes, labels, scores, and colored masks overlaid on the image.
- `compute_mask_iou(pred_mask, gt_mask)`: IoU between predicted and ground-truth masks.
- `evaluate_map(dataset)`: computes mean Average Precision (mAP) at IoU 0.5 for masks.
- Shows detection vs. mask quality comparison.

## Key Functions/Classes

| Component | Class/Function | Input → Output |
|---|---|---|
| Dataset | `ShapeDataset` | index → (image, targets) |
| Backbone | `SimpleBackbone` | (B,3,128,128) → (B,128,8,8) |
| RPN | `RPN` | features → (proposals, cls_loss, box_loss) |
| RoIAlign | `roi_align` | (features, rois) → (N, C, P, P) |
| Mask Head | `MaskHead` | (N, C, 14, 14) → (N, K+1, 28, 28) |
| Full Model | `MaskRCNN` | (images, targets) → losses/predictions |
| Loss | `MaskRCNNLoss` | predictions, targets → scalar loss |
| Eval | `compute_mask_iou`, `evaluate_map` | (pred, gt) → metrics |

## Data Flow / Shapes

```
Image (B, 3, 128, 128)
  → SimpleBackbone → (B, 128, 8, 8)              [stride 16]
  → RPN → proposals (N, 4) in image coords
  → roi_align(features, proposals, 7×7)  → (N, 128, 7, 7)   [box features]
  → roi_align(features, proposals, 14×14) → (N, 128, 14, 14)  [mask features]
  → BoxHead → class_logits (N, K+1), box_deltas (N, K+1, 4)
  → MaskHead → mask_logits (N, K+1, 28, 28) → sigmoid → binary masks
```

## Deliberate Simplifications vs. Full Paper

1. **Backbone:** Simple 4-block CNN instead of ResNet-50/101 or ResNeXt. Sufficient for synthetic shapes, not for COCO.
2. **Dataset:** Synthetic shapes (rect/circle/triangle) instead of COCO's 80 classes. Allows full training in minutes on 1 GPU without downloading data.
3. **No FPN:** Uses a single-level feature map instead of Feature Pyramid Network. FPN is critical for multi-scale object detection on real images but unnecessary when all shapes are the same scale range.
4. **No OHEM:** Hard example mining is omitted; all positive/negative RoIs contribute to loss. Simpler to implement.
5. **RPN and detection head share the backbone** but are trained jointly (the paper trains RPN separately for ablation convenience).
6. **Mask resolution:** 28×28 (matching paper) but the deconv architecture is simplified to 2 conv + 1 deconv + 1 output conv.
7. **No multi-scale training/testing:** Single image scale (128×128) instead of multi-scale pyramid.
8. **No keypoint branch:** Only instance segmentation is demonstrated; the human pose estimation extension is omitted.
