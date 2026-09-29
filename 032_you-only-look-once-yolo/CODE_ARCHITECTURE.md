# Code Architecture — YOLO Notebook

## Overview

The notebook implements a simplified YOLO object detector from scratch in PyTorch. It trains on a synthetic dataset of colored shapes (circles, rectangles, triangles) placed on plain backgrounds, predicting bounding boxes and class labels using the grid-based YOLO architecture.

## Section-by-Section Breakdown

### 1. Configuration & Hyperparameters

Defines all key constants:
- `S = 7` — grid size (7×7)
- `B = 1` — bounding boxes per cell (simplified from paper's B=2)
- `C = 3` — number of classes (circle, rectangle, triangle)
- `IMG_SIZE = 224` — input image resolution (reduced from 448 for speed)
- `LAMBDA_COORD = 4` — weight for localization loss
- `LAMBDA_NOOBJ = 0.5` — weight for no-object confidence loss
- `BATCH_SIZE = 32`, `EPOCHS = 25`, `LR = 1e-3`

### 2. Synthetic Dataset Generation

**Class: `ShapeDataset`**
- Generates random colored shapes on 224×224 backgrounds
- Each image contains 1-3 shapes with bounding boxes
- Returns: image tensor (3×224×224), target tensor (7×7×(5+3)=7×7×8)
- Target encoding per cell: [confidence, x, y, w, h, class_0, class_1, class_2]
  - x, y: offset from cell top-left, normalized to [0, 1] within the cell
  - w, h: normalized by image size, square-root applied during loss

**Key functions:**
- `__getitem__` — draws shapes using PIL, computes grid-cell assignment based on center
- `collate_fn` — standard batching

### 3. YOLO Network Architecture

**Class: `YOLONet`**
- **Backbone:** Simplified conv stack (6 conv layers with 1×1 reductions + 3×3 convs)
  - Conv 3→16 (3×3, stride 1) → Conv 16→32 (3×3, stride 2) → Conv 32→64 (1×1) → Conv 64→128 (3×3, stride 2) → Conv 128→256 (1×1) → Conv 256→512 (3×3, stride 2)
  - Leaky ReLU (0.1 slope) after each conv
  - Output spatial: 7×7×512
- **Detection head:** 
  - Flatten → FC(7*7*512 → 1024) → LeakyReLU → Dropout(0.5)
  - FC(1024 → 7*7*(5*B + C)) = FC(1024 → 7*7*8 = 392)
  - Reshape to (S, S, 8)
  - Sigmoid on final output (bounds to [0,1])

**Data flow:**
```
Input: (B, 3, 224, 224)
  → Conv backbone → (B, 512, 7, 7)
  → Flatten → (B, 25088)
  → FC → (B, 1024)
  → FC → (B, 392)
  → Reshape → (B, 7, 7, 8)
  → Sigmoid → predictions
```

### 4. YOLO Loss Function

**Class: `YOLOLoss`**

Multi-part sum-squared error loss, matching the paper's Equation 2:

1. **Coordinate loss** (λ_coord = 4):
   - For cells with objects: `(x - x̂)² + (y - ŷ)² + (√w - √ŵ)² + (√h - √ĥ)²`

2. **Confidence loss (object)**:
   - For cells with objects: `(confidence - IOU)²` where IOU is computed between predicted and true boxes
   - Simplified: target confidence = 1 for cells containing an object

3. **Confidence loss (no-object)** (λ_noobj = 0.5):
   - For cells without objects: `(confidence - 0)²` weighted by 0.5

4. **Classification loss**:
   - For cells with objects: sum over classes of `(prob_c - target_c)²`

**Mask computation:**
- `obj_mask`: 7×7 binary, 1 where an object center falls in that cell
- `noobj_mask`: inverse, but also down-weighted

### 5. Training Loop

- Standard PyTorch training loop
- Adam optimizer, lr=1e-3
- Prints loss components (coord, conf, cls) per epoch
- Stores loss history for plotting
- Saves model checkpoint

### 6. Evaluation & Visualization

- **Intersection-over-Union (IoU):** Standard box IoU computation
- **Non-Maximum Suppression:** Filters overlapping predictions by confidence threshold + IoU threshold
- **Visualization:** Draws predicted boxes on test images with class labels and confidence scores
- **Metrics:** Reports coordinate loss, confidence loss, classification loss trends

## Key Functions

| Function | Purpose |
|---|---|
| `compute_iou(box1, box2)` | Computes IoU between two bounding boxes |
| `non_max_suppression(boxes, scores, threshold)` | Removes overlapping low-confidence detections |
| `encode_target(boxes, labels, S, B, C)` | Converts raw annotations to YOLO grid format |
| `decode_prediction(pred, S, B, C, conf_thresh)` | Converts network output to bounding boxes + labels |
| `YOLOLoss.forward(pred, target)` | Computes multi-part YOLO loss |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Input size | 448×448 | 224×224 |
| Conv layers | 24 | 6 |
| Boxes per cell (B) | 2 | 1 |
| Classes (C) | 20 (PASCAL VOC) | 3 (synthetic shapes) |
| Dataset | PASCAL VOC 2007/2012 | Synthetic colored shapes |
| Pretraining | ImageNet (1 week) | None (from scratch) |
| Epochs | 120 | 25 |
| Output tensor | 7×7×30 | 7×7×8 |
| Two-stage training | Objectness first, then joint | Single-stage joint training |
