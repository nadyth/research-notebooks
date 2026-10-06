# Code Architecture — DeepWalk: Online Learning of Social Representations

## Notebook Overview

The notebook implements DeepWalk from scratch in three stages, then compares against node2vec's biased walks:

1. **Uniform random walk generation** — truncated random walks on a synthetic community graph
2. **Skip-gram with negative sampling** — from-scratch numpy implementation to learn node embeddings
3. **Comparison with node2vec** — reimplement node2vec's biased second-order walks on the same graph and compare embedding quality (NMI, silhouette, t-SNE visualization)

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `networkx` and `scikit-learn` (pip-installable, Colab/Kaggle compatible)
- Imports NumPy, NetworkX, scikit-learn (KMeans, t-SNE, silhouette, NMI), Matplotlib
- Sets random seeds for reproducibility

### 2. Graph Construction
- Builds a synthetic graph with 3 communities (each a small-world / clustered subgraph) connected by bridge edges
- This mirrors the structure of real social networks: tight communities with sparse inter-community links
- Shape: adjacency list representation; ~60-90 nodes total
- **Deliberate simplification:** The paper uses BlogCatalog (10,312 nodes), Flickr, and YouTube (1,138,499 nodes). We use a small synthetic graph so the full pipeline runs in under 5 minutes on CPU/GPU and embeddings can be visualized in 2D.

### 3. Uniform Random Walk Generation (DeepWalk core)
- `deepwalk_random_walk(G, start_node, walk_length)`: starting from a node, repeatedly move to a uniformly-random neighbor for `walk_length` steps
- `generate_deepwalk_walks(G, num_walks, walk_length)`: for each node, generate `num_walks` random walks
- Key property: walks are **uniform** — every neighbor is equally likely. No bias parameters.
- Output: list of walks (each a list of node IDs), treated as "sentences" of node IDs

### 4. Skip-gram with Negative Sampling (from scratch)
- `build_vocab(walks)`: maps node IDs to integer indices, computes unigram frequency table
- `skipgram_train(walks, vocab_size, dim, window_size, neg_samples, epochs, lr)`: 
  - For each walk, slide a context window of size `window_size`
  - For each (center, context) positive pair: update embeddings via gradient ascent on log σ(v_c · v_w)
  - Sample `neg_samples` negative nodes from unigram^0.75 distribution; push their dot products toward 0
  - Gradient updates: dL/dv_c = (1 - σ(v_c · v_w)) · v_w; dL/dv_w = (1 - σ(v_c · v_w)) · v_c
  - Learning rate decay: linear decay from `lr` to a small minimum
- **Deliberate simplification:** The paper uses Hierarchical Softmax for scalability to millions of nodes. We use negative sampling (simpler, equally effective on small graphs, and what word2vec popularized).

### 5. Biased Random Walks (node2vec, for comparison)
- `compute_alias_table(probs)`: alias sampling for O(1) sampling from a discrete distribution
- `biased_random_walk(G, start, walk_length, p, q, alias_tables)`: second-order biased walk where:
  - p=1, q=1 → equivalent to DeepWalk's uniform walk
  - p=1, q>1 → BFS-like (structural equivalence)
  - p=1, q<1 → DFS-like (homophily / community)
- `generate_node2vec_walks(G, num_walks, walk_length, p, q)`: generate walks with biased parameters
- Reuses the same Skip-gram training function from Section 4

### 6. Evaluation & Comparison
- **t-SNE visualization:** Project 16-dim embeddings to 2D, color by ground-truth community
- **K-Means clustering:** Run K-Means on embeddings, compare cluster assignments to ground truth
- **NMI (Normalized Mutual Information):** Quantitative measure of how well embeddings recover community structure
- **Silhouette Score:** Measures cluster cohesion/separation in embedding space
- Compare: DeepWalk (uniform) vs node2vec BFS-like (p=1, q=2) vs node2vec DFS-like (p=1, q=0.5)
- Expected: DeepWalk and DFS-like node2vec should both recover homophily-based communities well; BFS-like should capture structural roles differently

### 7. Graph Visualization with Learned Clusters
- Draw the original graph with nodes colored by K-Means clusters from DeepWalk embeddings
- Side-by-side with node2vec BFS-like and DFS-like clusterings
- Shows how different walk strategies lead to different community recovery patterns

### 8. Summary & Results
- Print NMI and silhouette scores for all three methods
- Discuss when DeepWalk (uniform) is sufficient vs when node2vec's bias helps
- Key takeaway: DeepWalk is the foundation; node2vec adds flexibility but DeepWalk's uniform walks are surprisingly competitive on homophily-dominated graphs

## Data Flow / Shapes

```
Graph G (NetworkX) 
  → generate_deepwalk_walks(G, num_walks=10, walk_length=20)
    → walks: list of lists of node IDs  [shape: (num_nodes * num_walks, walk_length)]
  → build_vocab(walks)
    → vocab: dict {node_id: int_index}, freq_table
  → skipgram_train(walks, vocab_size, dim=16, window_size=3, neg_samples=5, epochs=5)
    → embeddings: ndarray [shape: (vocab_size, 16)]
  → t-SNE(embeddings) → 2D points [shape: (vocab_size, 2)]
  → KMeans(n_clusters=3) → cluster labels [shape: (vocab_size,)]
  → NMI(cluster_labels, ground_truth_communities) → float
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Dataset | BlogCatalog, Flickr, YouTube (10K–1M nodes) | Synthetic 3-community graph (~75 nodes) |
| Walk params | γ=80 walks/node, t=40 length | γ=10 walks/node, t=20 length |
| Embedding dim | 128 | 16 |
| Negative samples | Not used (Hierarchical Softmax) | 5 negative samples |
| Window size | 10 | 3 |
| Optimization | Hierarchical Softmax + async SGD | Negative sampling + synchronous SGD |
| Downstream task | Multi-label node classification | Community recovery (NMI, K-Means) |
| GPU | Not required (CPU parallelism) | CPU-sufficient; GPU optional |

These simplifications keep the notebook runnable in under 5 minutes while preserving all the algorithmic essence of DeepWalk. The comparison with node2vec demonstrates the generalization relationship (DeepWalk = node2vec with p=q=1).
