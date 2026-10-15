# Code Architecture — Neural Collaborative Filtering

## Notebook Structure

### 1. Setup & Imports
- Install required packages (torch, numpy, pandas, matplotlib).
- Import PyTorch modules for model building.

### 2. Data Loading & Preprocessing
- Load MovieLens-100k (downloaded via URL or from sklearn-like utility).
- Convert to implicit feedback: binarize ratings (interacted = 1).
- Build user-item ID mappings (0-indexed contiguous).
- Create train/test split using leave-one-out: last item per user goes to test, rest to train.
- **Negative sampling:** For each positive interaction, sample 4 negatives.

### 3. Matrix Factorization (MF) Baseline
- **Class `MFModel`:** User embedding + Item embedding → dot product → sigmoid.
- Standard matrix factorization with learned embeddings.
- Trained with BCE loss on (positive + negative) pairs.
- Serves as the baseline to compare against.

### 4. Generalized Matrix Factorization (GMF)
- **Class `GMF`:** User GMF embedding + Item GMF embedding → element-wise multiply → linear layer → sigmoid.
- A neural version of MF where the element-wise product can be scaled by a learned weight vector.
- Architecture: `embed_dim → 1` via element-wise product + linear.

### 5. Multi-Layer Perceptron (MLP)
- **Class `MLP`:** User MLP embedding + Item MLP embedding → concatenate → stacked FC layers (ReLU) → sigmoid.
- Embedding dim: 32, concatenated to 64.
- Hidden layers: [64, 32, 16, 8] with ReLU activations.
- Output: single sigmoid score.

### 6. Neural Matrix Factorization (NeuMF)
- **Class `NeuMF`:** Combines GMF and MLP branches.
- GMF branch: separate user/item embeddings (dim 16) → element-wise product.
- MLP branch: separate user/item embeddings (dim 32) → concat → FC layers [64, 32, 16, 8].
- Concatenate GMF output (dim 16) and MLP output (dim 8) → final linear (dim 24 → 1) → sigmoid.
- **Pre-training option:** Initialize NeuMF with trained GMF and MLP weights (simplified: we skip full pre-training and train end-to-end for the notebook).

### 7. Training Loop
- **Function `train_model`:** Generic training loop accepting any model.
- BCE loss, Adam optimizer, learning rate 0.001.
- 10-15 epochs, batch size 256.
- Negative sampling regenerated each epoch (4:1 negative:positive ratio).
- Prints training loss per epoch.

### 8. Evaluation
- **Function `evaluate`:** Leave-one-out evaluation.
- For each user: score the held-out positive item + 99 randomly sampled negative items (100 candidates).
- Rank by predicted score.
- Metrics: Hit Ratio@10 (HR@10) and NDCG@10.
- **Data flow:** For each user → build candidate set → forward pass → sort scores → check if positive is in top-10.

### 9. Results Comparison
- Train MF, GMF, MLP, and NeuMF models.
- Print HR@10 and NDCG@10 for each model.
- Plot bar chart comparing all four models.
- Expected: NeuMF > MLP > GMF > MF (or similar ordering).

### 10. Discussion
- Print parameter counts for each model.
- Discuss why NeuMF combines linear (GMF) and non-linear (MLP) interactions.

## Key Functions/Classes

| Component | Purpose |
|-----------|---------|
| `MFModel` | Classic MF baseline (dot product) |
| `GMF` | Generalized MF (element-wise product + linear) |
| `MLP` | Multi-layer perceptron over concatenated embeddings |
| `NeuMF` | Fused model combining GMF + MLP |
| `train_model()` | Generic training loop with negative sampling |
| `evaluate()` | Leave-one-out HR@10 and NDCG@10 |
| `get_negatives()` | Sample negative items for training |
| `get_test_instances()` | Build 100-item candidate set per user for eval |

## Data Flow & Shapes

```
Input: (user_id, item_id) pairs → B×2 integers
  ├── GMF: user_emb[user_id] → B×16, item_emb[item_id] → B×16
  │         → element-wise multiply → B×16 → linear → B×1 → sigmoid
  ├── MLP: user_emb[user_id] → B×32, item_emb[item_id] → B×32
  │         → concat → B×64 → FC(64→32) → FC(32→16) → FC(16→8) → B×8
  └── NeuMF: concat(GMF_out B×16, MLP_out B×8) → B×24 → linear → B×1 → sigmoid

Output: B×1 predicted interaction probability
```

## Deliberate Simplifications vs Full Paper

1. **Dataset:** MovieLens-100k (943 users, 1682 items) instead of MovieLens-1M or Pinterest.
2. **Pre-training:** The paper pre-trains GMF and MLP separately before fusing into NeuMF. We train NeuMF end-to-end for simplicity (the paper notes end-to-end also works, just slightly less optimal).
3. **Negative sampling ratio:** 4:1 as in the paper, but we use simple random sampling instead of popularity-based sampling.
4. **Evaluation candidates:** 99 negative items + 1 positive = 100 per user (standard protocol).
5. **Embedding sizes:** Smaller than paper's (8/16/32 vs 64/128) to keep runtime under 30 minutes on Kaggle GPU.
6. **No dropout/regularization:** The paper uses dropout in some experiments; we omit it for simplicity on this small dataset.
