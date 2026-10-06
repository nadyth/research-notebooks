# Code Architecture — node2vec Notebook

## Overview

The notebook implements node2vec from scratch (no `node2vec` library, no `gensim` for the core algorithm) on a small synthetic graph with known community structure. It demonstrates biased second-order random walks, Skip-gram with negative sampling, and visualization of learned embeddings colored by community membership.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Uses `numpy`, `networkx`, `matplotlib`, `scikit-learn` (KMeans, t-SNE) — all pip-installable.
- No GPU required; runs entirely on CPU.

### 2. Graph Construction
- Builds a synthetic graph with 3 communities (Barabasi-Albert clusters connected by bridge edges).
- Each node is assigned a community label (0, 1, or 2).
- The graph is small (~60 nodes) for fast execution and clear visualization.
- Stores adjacency list, edge list, and node-to-community mapping.

### 3. Biased Random Walk Generation
- **`compute_transition_probs(graph, p, q)`**: Precomputes the 2nd-order transition probabilities. For each pair of (previous_node, current_node), computes α_pq for every neighbor of current_node based on shortest path distance from previous_node.
  - d(tx) = 0 → α = 1/p (return to previous node)
  - d(tx) = 1 → α = 1 (move to a neighbor of current that is also neighbor of previous)
  - d(tx) = 2 → α = 1/q (move outward to a node not adjacent to previous)
- **`alias_setup(probs)`**: Constructs the alias sampling tables (prob array and alias array) in O(n) for O(1) sampling.
- **`alias_draw(alias, prob)`**: Draws one sample in O(1) using the alias method.
- **`biased_random_walk(graph, start_node, walk_length, p, q)`**: Simulates one walk using the precomputed transition probabilities and alias sampling.
- **`generate_walks(graph, num_walks, walk_length, p, q)`**: Generates walks for all nodes, `num_walks` times each. Returns list of walk sequences (list of node-id lists).

### 4. Skip-gram with Negative Sampling
- **Vocabulary**: All nodes in the graph.
- **`skipgram_train(walks, dim, window_size, neg_samples, epochs, lr)`**: Implements Skip-gram with negative sampling from scratch using numpy.
  - Initializes embedding matrix W (|V| × dim) and context matrix C (|V| × dim).
  - For each walk, for each center node, samples context nodes within ±window_size.
  - Positive pair: (center, context) → sigmoid(W[center] · C[context]) → maximize log-likelihood.
  - Negative pairs: sample `neg_samples` random nodes → minimize sigmoid(W[center] · C[neg]).
  - Gradient updates with learning rate `lr`, using SGD.
  - Returns the embedding matrix W.

### 5. Visualization
- **t-SNE projection**: Reduces d-dimensional embeddings to 2D for visualization.
- **K-Means clustering**: Clusters embeddings into k=3 clusters to evaluate community recovery.
- **Scatter plot**: Colors nodes by true community label. Shows that nodes in the same community cluster together in embedding space.
- **Graph visualization**: Draws the original graph with node colors from learned cluster assignments.

### 6. Parameter Sensitivity (BFS vs DFS)
- Runs node2vec with two parameter settings:
  - p=1, q=0.5 → DFS-like (homophily): nodes in same community should cluster together.
  - p=1, q=2.0 → BFS-like (structural equivalence): nodes with similar roles should cluster together.
- Visualizes both to show the complementary embeddings, mirroring the Les Misérables case study from the paper.

### 7. Evaluation
- Computes clustering accuracy (after optimal label permutation) as a proxy for embedding quality.
- Prints the accuracy for both parameter settings.

## Key Functions/Classes

| Function | Purpose |
|---|---|
| `compute_transition_probs(G, p, q)` | Precompute 2nd-order transition probabilities |
| `alias_setup(probs)` | Build alias sampling tables |
| `alias_draw(alias, prob)` | O(1) sampling using alias method |
| `biased_random_walk(G, node, length, p, q, probs)` | Single biased walk |
| `generate_walks(G, n_walks, walk_len, p, q)` | All walks for all nodes |
| `skipgram_train(walks, dim, window, neg, epochs, lr)` | Skip-gram with negative sampling |
| `visualize_embeddings(emb, labels, title)` | t-SNE + scatter |
| `graph_with_clusters(G, emb, k, title)` | Graph viz with KMeans colors |

## Data Flow / Shapes

```
Graph (N nodes, E edges)
  → compute_transition_probs → dict of (prev, curr) → [(neighbor, α_pq), ...]
  → alias_setup → (prob_array, alias_array) per (prev, curr) node pair
  → generate_walks → list of walks: [[n0, n1, ...], [n0, n1, ...], ...]
  → skipgram_train → embedding matrix W: (N × d)
  → t-SNE → (N × 2) for visualization
  → KMeans → cluster assignments: (N,)
```

## Deliberate Simplifications vs Full Paper

1. **Graph size**: ~60 nodes instead of millions. The algorithm is identical; only scale differs.
2. **No parallelization**: Single-threaded. The paper emphasizes parallelism for scalability.
3. **Skip-gram implementation**: Hand-written numpy SGD instead of optimized C/word2vec. Slower but transparent.
4. **No edge feature learning**: We focus on node embeddings. The paper's edge operators (Hadamard, weighted-L1, etc.) are mentioned but not implemented.
5. **No semi-supervised parameter learning**: p and q are set manually. The paper shows they can be learned from a small labeled set.
6. **Negative sampling distribution**: Uniform random sampling instead of unigram^0.75 (no node degree frequency to weight by in a small graph).
