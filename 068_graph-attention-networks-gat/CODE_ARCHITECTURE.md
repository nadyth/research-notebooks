# Code Architecture — Graph Attention Networks (GAT)

## Notebook Overview

The notebook implements the Graph Attention Network (GAT) from Veličković et al. (2018) entirely from scratch using NumPy, then validates it on Zachary's Karate Club graph for semi-supervised node classification:

1. **Graph construction & preprocessing** — build the adjacency matrix, identify neighborhoods (including self-loops), prepare node features
2. **GAT attention layer from scratch** — implement the attention coefficient computation, masked softmax, and weighted neighbor aggregation
3. **Multi-head attention** — run K independent attention heads in parallel, concatenate (hidden) or average (output)
4. **Two-layer GAT training** — train end-to-end with cross-entropy loss on a few labeled nodes using gradient descent and manual backprop
5. **Evaluation & visualization** — classify all nodes, plot the graph with predicted vs true labels, and **visualize which neighbors receive the highest attention weights**

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `networkx` and `scikit-learn` (pip-installable, Colab/Kaggle compatible)
- Imports NumPy, NetworkX, scikit-learn (t-SNE), Matplotlib
- Sets random seeds for reproducibility

### 2. Graph Construction
- Loads Zachary's Karate Club graph from NetworkX (34 nodes, 78 edges, 2 communities)
- Extracts the adjacency matrix A (34×34), node features (identity matrix for one-hot node IDs), and ground-truth labels (2 communities)
- Builds the neighbor list (including self-loops): for each node, its neighbors + itself
- **Deliberate simplification:** The paper uses Cora (2,708 nodes, 7 classes), Citeseer, Pubmed, and PPI. We use the Karate Club graph so training completes in seconds and visualizations are interpretable. The math is identical — only the graph size differs.

### 3. GAT Attention Layer (from scratch)
- `gat_layer(X, W, a, neighbors, n_heads, activation, concat)`:
  - **Step 1 — Linear transform:** Project all node features: `H = X @ W` (shape: N × F')
  - **Step 2 — Attention coefficients:** For each node i and neighbor j, compute:
    `e_ij = LeakyReLU(a · [W·h_i || W·h_j])` 
    Implemented efficiently as: compute `a_src · H_i + a_dst · H_j` (splitting the attention vector `a` into source and destination parts), apply LeakyReLU(0.2)
  - **Step 3 — Masked softmax:** For each node i, softmax over only its neighbors' e_ij values (non-neighbors get -inf → 0 weight). This is the "masked" self-attention.
  - **Step 4 — Aggregation:** `h'_i = σ(Σ_j α_ij · W · h_j)` — weighted sum of transformed neighbor features, using attention weights α_ij
- **Self-loops:** Each node is included in its own neighbor list, so it attends to its own features alongside neighbors — no explicit A + I needed
- **Multi-head attention:** K independent sets of (W, a) run in parallel. Hidden layers concatenate outputs (shape: N × K*F'). Output layer averages (shape: N × F').

### 4. Two-Layer GAT Architecture
- **Layer 1:** K=4 heads, F'=8 features per head, ELU activation, output concatenated → (34 × 32)
- **Layer 2:** K=1 head, F'=2 output classes, softmax activation, output averaged → (34 × 2)
- Input X: (34 × 34) identity matrix
- **Deliberate simplification:** The paper uses K=8 heads and F'=8 in layer 1. We reduce to K=4 for faster training on the tiny Karate graph.

### 5. Loss & Backpropagation
- `cross_entropy(y_pred, y_true)`: Standard cross-entropy on labeled nodes only (semi-supervised)
- Manual gradient computation through attention, softmax, and linear layers:
  - Output layer gradient: `dL/dZ2 = (y_pred - y_onehot) / n_labeled` (softmax + CE)
  - Backprop through attention aggregation: `dL/dH_agg = α · dL/dZ2` (attention-weighted)
  - Backprop through linear transform: `dL/dW = H^T · dL/dH_agg`
  - Backprop through attention coefficients: requires differentiating through softmax and LeakyReLU
  - Hidden layer gradient through ELU activation
- Only labeled nodes contribute to the loss gradient, but all nodes participate in forward and gradient propagation
- **Deliberate simplification:** The paper uses Adam optimizer. We use vanilla gradient descent with a fixed learning rate for transparency and educational clarity.

### 6. Training Loop
- **Semi-supervised setup:** Label only a few nodes per class (e.g., 3 per community) — the "training mask"
- Gradient descent with learning rate 0.05, 300 epochs
- Track training accuracy and loss per epoch
- L2 weight decay applied to weight matrices W (λ = 5e-4)
- **Deliberate simplification:** The paper uses Adam, 200-700 epochs, and sparse dropout on features and attention. We use vanilla GD, no dropout (toy dataset converges without it).

### 7. Evaluation & Visualization
- **Node classification accuracy:** On all nodes, compare predicted class (argmax of output) with ground truth
- **Graph visualization:** Draw the karate club graph with node colors showing:
  - Ground-truth labels
  - Predicted labels
  - Side-by-side comparison highlighting misclassifications
- **Attention weight visualization (key deliverable):** For selected nodes, draw the graph with edge widths proportional to attention weights — showing which neighbors the model attends to most. This directly addresses the paper's key feature: interpretable, learned neighbor importance.
- **Embedding visualization:** Extract the 32-dim hidden layer representations, project to 2D with t-SNE, color by community — shows that GAT learns separable embeddings even with few labels
- **Multi-head attention comparison:** Show attention patterns from different heads — demonstrating that different heads learn different attention patterns (some focus on local structure, others on specific features)

## Data Flow / Shapes

```
X (34×34)  →  GAT Layer 1 (4 heads)  →  H1 (34×32)  →  GAT Layer 2 (1 head)  →  Z (34×2)  →  argmax → predictions (34,)
   ↓                                    ↓                                        ↓
 features    [head_k: α(34×34) @ (X@W_k)(34×8)]     α(34×34) @ (H1@W)(32→2)     softmax
              × 4 heads, concat                   × 1 head, average
```

- Input X: (N × F) = (34 × 34) — identity matrix (one-hot node features)
- Layer 1 per head: H_k = α_k @ (X @ W_k), W_k: (34×8), α_k: (34×34) attention matrix → (34×8)
- Layer 1 output: concat(H_1...H_4) → (34 × 32)
- Layer 2: Z = α @ (H1 @ W_out), W_out: (32×2) → (34×2)
- Attention matrices α_k: (34×34) — dense but effectively masked (non-neighbors = 0)

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|----------------|
| Dataset | Cora (2708 nodes, 7 classes) | Karate Club (34 nodes, 2 classes) |
| Features | 1433-dim bag-of-words | Identity matrix (one-hot node IDs) |
| Attention heads (layer 1) | K=8 | K=4 |
| Hidden features per head | F'=8 | F'=8 |
| Optimizer | Adam (lr=0.01) | Vanilla gradient descent (lr=0.05) |
| Epochs | 200-700 | 300 |
| Weight init | Glorot | Scaled random normal |
| Dropout | Sparse dropout on features + attention | Not applied (toy dataset) |
| L2 regularization | λ=5×10⁻⁴ | λ=5×10⁻⁴ (simplified weight decay) |
| Inductive evaluation | PPI (unseen test graphs) | Transductive only (Karate Club) |
| Implementation | TensorFlow | Pure NumPy (from scratch) |
