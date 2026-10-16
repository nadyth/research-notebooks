# Code Architecture — Deep Learning Recommendation Model (DLRM)

## Notebook Structure

### 1. Setup & Imports
- Install required packages (torch, numpy, pandas, matplotlib, scikit-learn).
- Import PyTorch modules: `nn`, `optim`, `DataLoader`, `Dataset`.
- Set random seed for reproducibility.
- Detect CUDA device (CPU fallback for local validation).

### 2. Data Loading & Preprocessing
- Load MovieLens-100k dataset (downloaded via URL).
- Construct a CTR-style binary classification problem: label = 1 if user rated item ≥ 4, else 0.
- Build categorical features: user_id, item_id (mapped to contiguous 0-indexed integers).
- Build dense/continuous features: extracted from rating timestamp (hour of day, day of week) and user statistics (mean rating, rating count) — normalized to [0, 1].
- Train/test split: 80/20 random split stratified by label.
- **Data shapes:**
  - Categorical features: `(batch, num_cat_features)` — integer indices
  - Dense features: `(batch, num_dense_features)` — float values
  - Labels: `(batch, 1)` — binary 0/1

### 3. DLRM Model Definition
- **Class `DLRM(nn.Module)`:**
  - **Embedding tables:** `nn.ModuleList` of `nn.Embedding` layers, one per categorical feature. Each maps vocabulary_size → embed_dim.
  - **Bottom MLP:** `nn.Sequential` of FC layers processing dense features. Architecture: `[num_dense, 128, 64, embed_dim]` with ReLU activations.
  - **Interaction layer:** Computes pairwise dot products between all processed vectors (embeddings + bottom MLP output). Takes upper triangular (i < j) to avoid redundancy and self-interactions. Returns `n*(n-1)/2` scalar values.
  - **Top MLP:** `nn.Sequential` of FC layers. Input = concatenation of interaction outputs + processed dense vector. Architecture: `[n*(n-1)/2 + embed_dim, 512, 256, 1]` with ReLU activations, final sigmoid.
- **Forward pass:**
  1. Embedding lookup: each categorical feature → embedding vector (batch, embed_dim)
  2. Bottom MLP: dense features → (batch, embed_dim)
  3. Collect all vectors: [emb_1, emb_2, ..., emb_n, dense_out]
  4. Interaction: pairwise dot products (upper triangular) → (batch, n*(n-1)/2)
  5. Concatenate: [interaction_output, dense_out] → (batch, n*(n-1)/2 + embed_dim)
  6. Top MLP → sigmoid → (batch, 1)

### 4. Neural Collaborative Filtering (NCF) Baseline for Comparison
- **Class `NCF(nn.Module)`:** Simplified NCF with user embedding + item embedding → concat → MLP → sigmoid.
- Architecture: user_emb (32d) + item_emb (32d) → concat (64d) → FC[64, 32, 16, 8] → sigmoid.
- Trained with BCE loss on the same binary classification task.
- This serves as the benchmark comparison required by the Code Template.

### 5. Training Loop
- **Optimizer:** SGD with momentum 0.9 (as in the DLRM paper) or Adam.
- **Loss:** BCEWithLogitsLoss (numerically stable).
- **Batch size:** 512.
- **Epochs:** 10 (small-scale for demonstration; paper uses much longer training).
- **Metrics:** AUC-ROC and accuracy on test set.
- Training loop: forward pass → loss → backward → optimizer step. Print loss every epoch.
- After training, evaluate both DLRM and NCF on the test set.

### 6. Evaluation & Comparison
- Compute test AUC-ROC and accuracy for both DLRM and NCF.
- Print comparison table.
- Plot training loss curves for both models.
- Plot AUC comparison bar chart.

### 7. Discussion
- Summary of how DLRM's explicit dot-product interactions differ from NCF's MLP-based interactions.
- Note on scaling: real DLRM uses embedding tables with millions of rows; our demo uses small vocabularies.
- Note on the bottom MLP: processes continuous features to the same dimension as embeddings, enabling fair dot-product interactions.

## Key Functions/Classes

| Component | Class/Function | Description |
|---|---|---|
| DLRM | `DLRM(nn.Module)` | Full DLRM model with embeddings, bottom MLP, interactions, top MLP |
| NCF Baseline | `NCF(nn.Module)` | Simplified NCF for comparison |
| Interaction | `interact_pairs(vectors)` | Computes upper-triangular pairwise dot products |
| Data prep | `prepare_data()` | Loads ML-100k, builds features, splits train/test |
| Training | `train_model()` | Generic training loop for any model |
| Evaluation | `evaluate_model()` | Computes AUC-ROC and accuracy |

## Data Flow / Shapes

```
Input:
  categorical: (B, num_cat)    e.g., (512, 2) for [user_id, item_id]
  dense: (B, num_dense)        e.g., (512, 4) for [hour, day_of_week, user_mean_rating, user_rating_count]

DLRM forward:
  embeddings: [(B, embed_dim)] × num_cat
  bottom_mlp_out: (B, embed_dim)
  all_vectors: (B, num_cat + 1, embed_dim)
  interactions: (B, (num_cat+1)*num_cat/2)    e.g., (512, 3) for 3 vectors → 3 pairs
  top_input: (B, num_interactions + embed_dim)
  output: (B, 1)  → sigmoid → probability

NCF forward:
  user_emb: (B, 32)
  item_emb: (B, 32)
  concat: (B, 64)
  MLP: (B, 64) → (B, 32) → (B, 16) → (B, 8)
  output: (B, 1) → sigmoid → probability
```

## Deliberate Simplifications vs Full Paper

1. **Scale:** Real DLRM uses Criteo dataset with millions of users/items and hundreds of categorical features. We use MovieLens-100k with 2 categorical features for tractability.
2. **Embedding table sizes:** Paper uses tables with millions of rows. Ours use actual vocabulary sizes from ML-100k (~943 users, ~1682 items).
3. **Parallelization:** Paper describes model-parallel embeddings + data-parallel MLPs across multiple GPUs. We run on a single device (GPU or CPU).
4. **Feature count:** Paper uses dozens of categorical + continuous features. We use 2 categorical + 4 continuous features.
5. **Training duration:** Paper trains for hours/days on full Criteo. We train 10 epochs for demonstration.
6. **Interaction layer:** Paper uses `nn.EmbeddingBag` for multi-hot lookups. We use simple `nn.Embedding` for single-valued categorical features.
7. **NCF comparison:** We implement a simplified NCF (MLP-only, no GMF/NeuMF fusion) rather than the full NeuMF model, to keep the comparison focused on interaction mechanism differences.
