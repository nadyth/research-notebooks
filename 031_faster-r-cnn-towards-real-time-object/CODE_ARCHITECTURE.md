# Faster R-CNN — Notebook Architecture

## Goal

The notebook implements a simplified Faster R-CNN pipeline from scratch in PyTorch: a small CNN backbone producing a shared feature map, a **Region Proposal Network (RPN)** with anchor boxes that predicts objectness and box regressions, **RoI (Region of Interest) pooling** to extract fixed-size features from proposals, and a **detection head** that classifies and refines proposals. It trains on a synthetic toy object-detection dataset (simple shapes — squares and circles — placed on blank canvases) and visualizes predicted bounding boxes.

## Section-by-section breakdown

### 1. Setup and imports
- Imports `torch`, `torch.nn`, `torchvision`, `matplotlib`, `numpy`.
- Sets device to CUDA if available, else CPU.
- Installs `torchvision` if missing.

### 2. Synthetic detection dataset
- Generates toy images (64×64 grayscale) containing 1–3 objects (squares or circles) at random positions with random sizes.
- Each object has: bounding box (x1, y1, x2, y2), class label (0=square, 1=circle).
- `ToyDetectionDataset`: returns image tensor and a list of (bbox, label) targets.
- This synthetic dataset avoids downloading large detection datasets (PASCAL VOC, COCO) and keeps training fast while demonstrating all pipeline components.

### 3. Backbone network
- A small CNN: 3 conv layers (32→64→128 channels) with ReLU and max-pooling.
- Input: 64×64 → feature map of 8×8 with 128 channels.
- This replaces VGG-16/ZF-Net from the paper for feasibility.

### 4. Anchor generation
- Generates anchors at each feature-map location (8×8 = 64 locations).
- 3 scales (8, 16, 24 pixels) × 3 aspect ratios (0.5, 1.0, 2.0) = 9 anchors per location → 576 total anchors.
- `generate_anchors()` returns anchor boxes in (x1, y1, x2, y2) format mapped to original image coordinates.

### 5. Region Proposal Network (RPN)
- A 3×3 conv on the feature map (512 intermediate channels) followed by two 1×1 conv heads:
  - **Objectness branch**: 1×1 conv → 2×k channels (foreground/background per anchor).
  - **Box regression branch**: 1×1 conv → 4×k channels (dx, dy, dw, dh per anchor).
- `RPNHead` module with forward returning cls_logits and bbox_deltas.
- RPN loss: classification (cross-entropy on objectness) + smooth L1 regression on matched anchors.

### 6. Anchor labeling and matching
- `assign_targets_to_anchors()`: IoU-based matching.
  - Anchors with IoU ≥ 0.7 → positive (object).
  - Anchors with IoU < 0.3 → negative (background).
  - Best-IoU anchor per GT box forced positive.
- Encoding/decoding box deltas using the parameterization from the paper: dx = (x - xa)/wa, etc.

### 7. Proposal generation (NMS)
- `apply_box_deltas()`: decodes RPN predictions to image-space boxes.
- `nms()`: non-maximum suppression to reduce overlapping proposals to top-N (e.g., top 50 for training, top 300 for inference).
- During training, sample 32 positive + 32 negative anchors for the RPN loss.

### 8. RoI Pooling
- `roi_pool()`: extracts fixed-size (7×7) features from the shared feature map for each proposal.
- Uses adaptive max pooling on the feature map region corresponding to each proposal's box coordinates (scaled to feature-map resolution).
- Output: (N_proposals, 128, 7, 7) tensor fed to the detection head.

### 9. Detection head (Fast R-CNN)
- Flatten RoI-pooled features → FC(128*7*7, 256) → ReLU → FC(256, 256) → ReLU.
- Two output heads:
  - **Classification**: FC(256, num_classes + 1) — includes background class.
  - **Box regression**: FC(256, (num_classes + 1) * 4) — class-specific box refinement.
- Detection loss: cross-entropy classification + smooth L1 regression on matched proposals.

### 10. Full model and training loop
- `FasterRCNN` module combining backbone + RPN + RoI pool + detection head.
- End-to-end training: each iteration samples images, runs backbone → RPN → proposals → RoI pool → detection head, computes combined loss (RPN cls + RPN reg + det cls + det reg), backpropagates.
- Adam optimizer, learning rate 1e-3, 5 epochs on the toy dataset.

### 11. Visualization
- Draws predicted bounding boxes on sample test images with class labels and confidence scores.
- Shows RPN proposals before and after NMS.
- Plots training loss curves (RPN loss, detection loss, total).

## Key functions / classes

| Name | Purpose |
|------|---------|
| `ToyDetectionDataset` | Synthetic image generator with shape objects + bboxes |
| `Backbone` | Small CNN producing shared feature map |
| `generate_anchors()` | Creates anchor boxes at all feature-map locations |
| `RPNHead` | 3×3 conv → objectness + regression heads |
| `assign_targets_to_anchors()` | IoU-based anchor-GT matching |
| `encode_boxes()` / `decode_boxes()` | Box parameterization (dx, dy, dw, dh) |
| `nms()` | Non-maximum suppression for proposals |
| `roi_pool()` | Fixed-size feature extraction from proposals |
| `DetectionHead` | FC classifier + regressor over RoI features |
| `FasterRCNN` | Full model combining all components |
| `compute_rpn_loss()` | RPN classification + regression loss |
| `compute_detection_loss()` | Detection classification + regression loss |

## Data flow / shapes

```
Input image: (B, 1, 64, 64)
  → Backbone: (B, 128, 8, 8)         # shared feature map
  → RPN: cls_logits (B, 576, 2), bbox_deltas (B, 576, 4)
  → NMS: top-50 proposals (50, 4)
  → RoI Pool: (50, 128, 7, 7)
  → Flatten + FC: (50, 256)
  → Det head: cls_logits (50, 3), bbox_pred (50, 12)
```

## Deliberate simplifications vs. the full paper

1. **Backbone**: 3-layer CNN instead of VGG-16 (16 layers) or ZF-Net — enough to learn features on toy shapes.
2. **Image size**: 64×64 grayscale instead of 1000×600 RGB.
3. **Classes**: 2 toy classes (square, circle) + background instead of 20 PASCAL VOC classes.
4. **Anchors**: 9 per location (same as paper) but only 64 locations (8×8 feature map).
5. **No alternating training**: The paper alternates RPN and Fast R-CNN training; we do joint end-to-end training for simplicity.
6. **No image pyramid**: Single-scale detection only.
7. **Proposal count**: Top-50 during training (paper uses 2000/300) for speed.
8. **RoI Pooling**: Simple adaptive max pool instead of the exact RoI pooling with bin subdivision.
