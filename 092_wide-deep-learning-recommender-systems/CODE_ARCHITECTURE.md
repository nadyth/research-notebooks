# Code Architecture — Wide & Deep Learning for Recommender Systems

## Notebook Structure

### 1. Setup & Imports
- Install required packages (torch, numpy, pandas, scikit-learn, matplotlib).
- Import PyTorch modules for model building.
- Set random seeds for reproducibility.

### 2. Data Generation & Preprocessing
- Build a **synthetic recommendation dataset** that mimics the app-install scenario:
  - Users with categorical features (country, device_type, age_group).
  - Items (apps) with categorical features (app_category, app_price_tier).
  - Target: binary app acquisition label (install = 1, no install = 0).
- The synthetic data includes a **memorizable pattern** (specific user-item cross-feature co-occurrences strongly predict install) and a **generalizable pattern** (embedding-based similarity predicts install).
- Vocabulary mapping: each categorical feature string mapped to an integer ID.
- Train/test split: 80/20 stratified by label.

### 3. Feature Engineering — Wide Component
- **Cross-product features:** Manually construct AND combinations of key categorical features, e.g.:
  - `AND(user_country=US, app_category=games)`
  - `AND(device_type=phone, app_price_tier=free)`
  - `AND(age_group=young, app_category=social)`
- These cross-product features are one-hot encoded and concatenated with raw one-hot features to form the wide input vector.
- The wide input is a sparse binary vector of raw features + cross-product features.

### 4. Model Definitions

#### 4a. WideModel (Wide-Only Baseline)
- **Class `WideModel`:** Linear layer over the wide feature vector (raw + cross-product features).
- Input: sparse binary wide features → linear → sigmoid.
- This is the generalized linear model with cross-product transformations from the paper.

#### 4b. DeepModel (Deep-Only Baseline)
- **Class `DeepModel`:** Embedding tables for each categorical feature → concatenate embeddings + dense features → stacked ReLU hidden layers → sigmoid.
- Each categorical feature has its own embedding table (dimension 16).
- Embeddings concatenated with normalized continuous features → 3 hidden layers (128, 64, 32) with ReLU.
- This is the feed-forward neural network with embeddings from the paper.

#### 4c. WideDeepModel (Joint Model)
- **Class `WideDeepModel`:** Combines WideModel and DeepModel.
- Wide part: same linear layer over cross-product features → wide_logit.
- Deep part: same embedding + MLP → deep_logit.
- Fusion: `sigma(w_wide * wide_logit + w_deep * deep_logit + b)` — learned fusion weights.
- All parameters trained jointly via backpropagation through a shared BCE loss.

### 5. Training Loop
- **Function `train_model`:** Generic training loop for any model.
- BCE loss, Adam optimizer (simplification: paper used FTRL for wide, AdaGrad for deep; we use Adam for all for simplicity).
- 20 epochs, batch size 256, learning rate 0.001.
- Prints training loss and AUC per epoch.
- Early stopping on validation AUC plateau.

### 6. Evaluation
- **Metrics:** AUC (Area Under ROC Curve), Accuracy, Log Loss.
- Evaluate on held-out test set.
- **Function `evaluate_model`:** Forward pass on test data, compute predictions, calculate AUC and accuracy via sklearn metrics.

### 7. Train & Evaluate All Three Models
- Train WideModel (wide-only baseline).
- Train DeepModel (deep-only baseline).
- Train WideDeepModel (joint model).
- Print training curves and test metrics for each.

### 8. Results Comparison
- Print comparison table: Model | AUC | Accuracy | Log Loss | # Parameters.
- Plot ROC curves for all three models on the same figure.
- Plot training loss curves for all three models.
- Expected: WideDeep > Deep > Wide (or WideDeep best, with Wide strong on memorizable patterns and Deep strong on generalizable patterns).

### 9. Analysis — Memorization vs Generalization
- **Memorization test:** Evaluate on test examples where the cross-product pattern is present (seen during training). Wide model should excel here.
- **Generalization test:** Evaluate on test examples with unseen feature combinations. Deep model should excel here.
- Show that Wide & Deep performs well on BOTH, demonstrating the complementary strengths.

### 10. Parameter Count Comparison
- Print parameter counts for all three models.
- Discuss how joint training keeps the wide part small (only needs to complement deep part) vs ensemble where each model must be full-size.

### 11. Discussion
- Key findings: Wide & Deep outperforms both baselines.
- The wide part captures specific cross-feature rules; the deep part generalizes.
- Joint training vs ensemble distinction.
- Simplifications and real-world considerations.

## Key Functions/Classes

| Component | Purpose |
|-----------|---------|
| `SyntheticRecData` | Generate synthetic recommendation dataset with memorizable + generalizable patterns |
| `WideModel` | Linear model over raw + cross-product features (wide-only baseline) |
| `DeepModel` | Embedding + MLP model (deep-only baseline) |
| `WideDeepModel` | Joint wide + deep model with learned fusion |
| `train_model()` | Generic training loop with BCE loss |
| `evaluate_model()` | Compute AUC, accuracy, log loss on test set |
| `build_wide_features()` | One-hot encode raw + cross-product features |

## Data Flow & Shapes

```
Input: (user_features, item_features) per example
  |
  ├── WIDE: raw one-hot + cross-product one-hot → sparse vector (dim ~200)
  │         → linear layer → wide_logit (scalar)
  |
  ├── DEEP: categorical features → embedding tables → concat (dim ~80)
  │         + dense features (dim ~3) → concat (dim ~83)
  │         → FC(83→128) → ReLU → FC(128→64) → ReLU → FC(64→32) → ReLU
  │         → FC(32→1) → deep_logit (scalar)
  |
  └── FUSION: w_wide * wide_logit + w_deep * deep_logit + b → sigmoid → P(install)

Output: scalar predicted install probability
```

## Deliberate Simplifications vs Full Paper

1. **Dataset:** Synthetic recommendation dataset (~10K examples) instead of Google Play's 500 billion examples. The synthetic data is designed with both memorizable and generalizable patterns to demonstrate the paper's core claim.
2. **Optimizer:** Adam for all components instead of FTRL Proximal (wide) + AdaGrad (deep). FTRL with L1 produces sparser wide weights, but Adam works fine for this scale.
3. **Cross-product features:** Manually defined 6-8 key cross-product combinations instead of the full automated cross-product enumeration. The paper's Google Play deployment used cross-products of user_installed_app × impression_app.
4. **Embedding size:** 16-dimensional embeddings (paper used 32). Sufficient for our small vocabulary.
5. **Deep layers:** 3 hidden layers [128, 64, 32] (paper used 3 ReLU layers with ~1200 input dim). Scaled down for our smaller feature space.
6. **No warm-starting:** The paper describes a warm-starting system for incremental retraining. We train from scratch each time.
7. **No serving optimization:** We don't implement multithreaded batch inference. The paper's 14ms latency optimization is an engineering detail beyond the model architecture.
8. **Fusion weights:** We learn scalar fusion weights (w_wide, w_deep) rather than the paper's vector-level weighting, which is sufficient at our scale.
