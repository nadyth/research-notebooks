# Code Architecture — Semi-Supervised Classification with Graph Convolutional Networks

## Notebook Overview

The notebook implements the Graph Convolutional Network (GCN) from Kipf & Welling (2017) entirely from scratch using NumPy, then validates it on Zachary's Karate Club graph for semi-supervised node classification:

1. **Graph construction & preprocessing** — build the adjacency matrix, add self-loops, compute the symmetric-normalized propagation matrix D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ²
2. **GCN layer from scratch** — implement the propagation rule H' = σ(D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² H W) as a NumPy matrix operation
3. **Two-layer GCN training** — train end-to-end with cross-entropy loss on a few labeled nodes, using gradient descent and manual backprop
4. **Evaluation & visualization** — classify all nodes, plot the graph with predicted vs true labels, visualize learned embeddings via t-SNE

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `networkx` and `scikit-learn` (pip-installable, Colab/Kaggle compatible)
- Imports NumPy, NetworkX, scikit-learn (t-SNE), Matplotlib
- Sets random seeds for reproducibility

### 2. Graph Construction
- Loads Zachary's Karate Club graph from NetworkX (34 nodes, 78 edges, 2 communities)
- Extracts the adjacency matrix A (34×34), node features (identity matrix or degree-based), and ground-truth labels
- **Deliberate simplification:** The paper uses Cora (2,708 nodes, 5,429 edges, 7 classes), Citeseer, Pubmed, and NELL. We use the Karate Club graph so training completes in seconds and visualizations are interpretable. The math is identical — only the graph size differs.

### 3. Normalized Adjacency Matrix (Key Preprocessing)
- `normalized_adjacency(A)`: Computes Ã = A + I (self-loops), then D̃ = degree matrix of Ã, then Â = D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ²
- The symmetric normalization ensures that high-degree nodes don't dominate aggregation
- This single matrix Â is the entire "graph convolution" — applied at every layer via matrix multiply
- Shape: Â is (N×N) where N=34; sparse in practice but dense here for simplicity

### 4. GCN Layer (from scratch)
- `gcn_layer(H, W, A_norm, activation)`: Computes Z = A_norm @ H @ W, then applies activation (ReLU for hidden, softmax for output)
- Parameters: W^(0) is (F_in × 16), W^(1) is (16 × C) where C=2 classes
- Forward pass: H1 = ReLU(A_norm @ X @ W0), H2 = softmax(A_norm @ H1 @ W1)
- **Deliberate simplification:** The paper uses weight initialization based on Glorot & Bengio (2010). We use simple random normal initialization scaled by input dimension.

### 5. Loss & Backpropagation
- `cross_entropy(y_pred, y_true)`: Standard cross-entropy on labeled nodes only
- Manual gradient computation through both layers:
  - dL/dZ2 = (y_pred - y_onehot) / n_labeled  (softmax + CE gradient)
  - dL/dH1 = A_norm.T @ (dL/dZ2 @ W1.T)  (gradient through propagation)
  - dL/dW1 = H1.T @ (A_norm @ dL/dZ2)   (gradient through linear layer)
  - ReLU mask: zero out gradients where H1 ≤ 0
  - dL/dH0 = A_norm.T @ (ReLU_mask * (dL/dH1 @ W0.T))
  - dL/dW0 = X.T @ (A_norm @ dL/dH1_masked)
- Only labeled nodes contribute to the loss gradient, but all nodes participate in the forward pass and gradient propagation

### 6. Training Loop
- **Semi-supervised setup:** Label only a few nodes per class (e.g., 2-3 per community) — the "training mask"
- Gradient descent with learning rate 0.01, 200 epochs
- Track training accuracy and loss per epoch
- **Deliberate simplification:** The paper uses Adam optimizer and trains for 200-1000 epochs. We use vanilla gradient descent with a fixed learning rate for transparency.

### 7. Evaluation & Visualization
- **Node classification accuracy:** On all nodes, compare predicted class (argmax of output) with ground truth
- **Graph visualization:** Draw the karate club graph with node colors showing:
  - Ground-truth labels
  - Predicted labels
  - Side-by-side comparison highlighting misclassifications
- **Embedding visualization:** Extract the 16-dim hidden layer representations, project to 2D with t-SNE, color by community — shows that GCN learns separable embeddings even with few labels
- **Label propagation effect:** Show how classification accuracy improves as we vary the number of labeled nodes (1 to 10 per class), demonstrating the semi-supervised advantage

## Data Flow / Shapes

```
X (34×34)  →  GCN Layer 1  →  H1 (34×16)  →  GCN Layer 2  →  Z (34×2)  →  argmax → predictions (34,)
   ↓              ↓                               ↓
 features    Â(34×34)@X(34×34)@W0(34×16)    Â(34×34)@H1(34×16)@W1(16×2)
              + ReLU                          + softmax
```

- Input X: (N × F) = (34 × 34) — identity matrix (one-hot node features)
- Hidden H1: (N × 16) — 16 hidden units
- Output Z: (N × C) = (34 × 2) — 2 classes (the two karate club factions)
- Normalized adjacency Â: (N × N) = (34 × 34) — precomputed, reused at every layer

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|----------------|
| Dataset | Cora (2708 nodes, 7 classes) | Karate Club (34 nodes, 2 classes) |
| Features | 1433-dim bag-of-words | Identity matrix (one-hot node IDs) |
| Optimizer | Adam | Vanilla gradient descent |
| Epochs | 200-1000 | 200 |
| Weight init | Glorot/Bengio | Scaled random normal |
| Dropout | Applied to features and layer | Not applied (toy dataset converges without it) |
| Spectral theory | Full derivation from Chebyshev polynomials | Direct implementation of the simplified propagation rule |
| Implementation | TensorFlow | Pure NumPy (from scratch) |
