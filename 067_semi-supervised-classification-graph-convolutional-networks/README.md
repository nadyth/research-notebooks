# Semi-Supervised Classification with Graph Convolutional Networks (GCN)

**Paper:** Thomas N. Kipf, Max Welling. *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR 2017. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/semi-supervised-classification-graph-conv)

## Summary

Graph Convolutional Networks (GCNs) by Kipf & Welling introduced a simple, scalable, and highly influential architecture for semi-supervised node classification on graph-structured data. The core innovation is a **first-order approximation of spectral graph convolutions** that reduces the computationally expensive eigendecomposition of the graph Laplacian to a simple **normalized adjacency matrix propagation** step. Each GCN layer updates node representations by aggregating (weighted) features from immediate neighbors and applying a learned linear transformation followed by a nonlinearity — the propagation rule is **H' = σ(D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² H W)**, where Ã = A + I is the adjacency matrix with self-loops, D̃ is its degree matrix, H is the node feature matrix, and W is a learnable weight matrix. This elegant formulation scales linearly in the number of edges, requires no eigenvalue computation, and naturally encodes both node features and graph topology. The paper demonstrates that a 2-layer GCN with just a few labeled nodes per class achieves state-of-the-art results on citation networks (Cora, Citeseer, Pubmed) and a knowledge graph (NELL), outperforming methods that rely on graph-based regularization or random walks.

### Core Idea

The key insight is that spectral graph convolutions — defined in the Fourier domain via the eigendecomposition of the graph Laplacian L = UΛUᵀ — can be approximated by truncating the Chebyshev polynomial expansion to first order. This yields a filter that depends only on the normalized adjacency matrix, eliminating the need to compute any eigenvectors. The resulting layer performs a **localized first-order neighborhood aggregation**: each node gathers features from its direct neighbors (plus itself via self-loops), weights them by node degree (symmetric normalization), and passes the result through a learned linear projection. Stacking two such layers gives a receptive field of 2 hops, which turns out to be sufficient for many node-classification tasks. Semi-supervised learning is achieved by training the GCN end-to-end with cross-entropy loss on the small set of labeled nodes; the graph structure acts as a built-in inductive bias that propagates label information to unlabeled neighbors.

### Key Method Details

- **Graph convolution layer:** H^(l+1) = σ(D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² H^(l) W^(l)). The matrix D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² is precomputed once; each forward pass is a sparse matrix multiplication — O(|E|·d) where |E| is the number of edges and d the feature dimension.
- **Self-loop augmentation:** Ã = A + I ensures each node aggregates its own features along with its neighbors'. Without self-loops, a node's own representation would be overwritten by its neighbors' in every layer.
- **Symmetric normalization:** D̃⁻¹ᐟ² Ã D̃⁻¹ᐟ² scales each neighbor's contribution by 1/√(deg(j)·deg(i)), preventing high-degree nodes from dominating and stabilizing training.
- **Two-layer architecture:** The paper's best models use just two GCN layers (16 hidden units → C output classes). The receptive field is 2 hops — surprisingly effective because most label homophily is local.
- **Semi-supervised cross-entropy:** Only labeled nodes contribute to the loss, but all nodes participate in forward propagation. The graph structure propagates information from labeled to unlabeled nodes.
- **Toy graph dataset (Karate Club / synthetic):** We use Zachary's Karate Club graph (34 nodes, 2 communities) or a small synthetic community graph, with a tiny feature matrix, so the entire pipeline trains in seconds and visualizations are clear.
- **Relation to spectral theory:** The full spectral convolution gθ ⋆ x = U gθ(Λ) Uᵀx requires O(N³) eigendecomposition. The Chebyshev approximation (Defferrard et al., 2016) reduces this to O(K|E|). Kipf & Welling further simplify to K=1 with a single parameter per filter, yielding the linear propagation rule above — the simplest possible spectral filter that still captures local structure.

### What problem does it solve?

Imagine you're a teacher with a big classroom of students, and you only have time to read the essays of a few of them — maybe 5 out of 30. But you notice that students who sit together and chat tend to write about similar topics. Can you guess what the rest of the class wrote about, just from those 5 essays and the seating chart?

That's exactly what GCN solves. It looks at a network — like a social network where people are connected if they're friends — and combines two sources of information: (1) what each person is "about" (their features, like their essay), and (2) who they're connected to (the friendship links). Even if you only label a few people (say, "this person likes sports"), the GCN passes that information along the friendship links. Since friends tend to share interests, the label spreads to neighbors, and neighbors of neighbors, automatically. After just two rounds of this message-passing, the network can confidently guess the interests of everyone — even the people whose essays you never read.

This is incredibly useful in real life: predicting what products someone will like based on their friends' purchases, classifying proteins by function based on their interaction networks, or recommending papers based on citation links — all with very few labeled examples.

### Influence

With over 40,000+ citations, the GCN paper is one of the most influential works in graph representation learning and geometric deep learning. It established the message-passing paradigm that underpins virtually all modern graph neural networks: GraphSAGE (Hamilton et al., 2017), GAT (Veličković et al., 2018), GIN (Xu et al., 2019), and the broader family of MPNNs. The first-order spectral approximation became the standard "graph conv" layer in PyTorch Geometric and DGL. The paper also inspired theoretical analyses of over-smoothing (Li et al., 2018), expressiveness bounds (Morris et al., 2019), and scaling laws for GNNs. Kipf & Welling later authored the widely-cited "Variational Graph Auto-Encoders" (2016) extending GCNs to unsupervised link prediction. The combination of spectral theory with practical deep learning made graph neural networks accessible to the broader ML community for the first time.

---

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/semi-supervised-classification-graph-conv)
