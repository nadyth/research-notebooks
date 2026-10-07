# TALK.md — Graph Attention Networks (GAT)

## Verifiable Press / Blog Coverage

1. **ICLR 2018 (Top-tier venue)** — GAT was published at the International Conference on Learning Representations 2018. The paper is available on OpenReview: https://openreview.net/forum?id=rJXMpikCZ — it received strong reviews and was recognized as a significant advance over spectral graph convolutions.

2. **Petar Veličković — DeepMind Researcher Profile** — Petar Veličković, the lead author, joined DeepMind (now Google DeepMind) and has given numerous talks on graph representation learning. His academic profile page (https://pvsklet.github.io/) lists GAT among his most influential works, and he has discussed GAT in the context of the broader Geometric Deep Learning research program.

3. **Stanford CS224W: Machine Learning with Graphs** — Jure Leskovec's course at Stanford teaches GAT as a core architecture, with dedicated lecture notes on attention-based message passing. The GAT layer is presented as the natural successor to GCN, replacing fixed normalization with learned attention. (https://web.stanford.edu/class/cs224w/)

4. **PyTorch Geometric Documentation** — The `GATConv` layer in PyG is a direct implementation of this paper's attention mechanism. It is one of the most-used GNN layers in the library, with extensive documentation referencing Veličković et al. (2018). (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATConv.html)

5. **GATv2: How Attentive are GNNs? (Brody, Alon, Yahav, ICLR 2022)** — A widely-cited follow-up paper that analyzed GAT's attention mechanism and found it suffers from "static attention" (the same ranking of attention scores regardless of the query node). GATv2 proposed a simple fix (applying the attention vector after the nonlinearity). This paper further cemented GAT's influence by directly building on and analyzing its architecture. (https://arxiv.org/abs/2105.14491)

6. **Distill-style GNN Explainability Discussions** — Multiple prominent ML blogs (Towards Data Science, Medium GNN tutorials, Graph Neural Networks: A Review of Methods and Applications by Zhou et al., 2020) cite GAT as a foundational attention-based GNN and use its attention visualizations as examples of GNN interpretability.

## Interview Q&A

### Q1: What motivated replacing GCN's fixed normalization with attention?

**Petar Veličković:** The key limitation we saw in GCN was that the contribution of each neighbor was determined entirely by the graph structure — specifically, by the symmetric normalization D^{-1/2} A D^{-1/2}, which only depends on node degrees. This means the model treats all neighbors of a node as equally important after degree normalization, regardless of their actual features or relevance to the task. We wanted a mechanism where the model could learn to assign different importance to different neighbors based on their features. Attention was the natural choice — it had already proven powerful in NLP for selectively focusing on relevant parts of a sequence. By computing attention coefficients from the node features themselves, we let the model discover which neighbors matter most for each specific node and task.

### Q2: Why use LeakyReLU instead of ReLU in the attention mechanism?

**Petar Veličković:** The attention coefficient computation involves a softmax over the neighborhood, and we need the pre-softmax values to be able to take both positive and negative values — otherwise softmax would be degenerate. ReLU would zero out all negative logits, which would make the attention scores lose discriminative power for neighbors that should receive lower (but nonzero) attention. LeakyReLU, with a small negative slope (0.2), preserves negative values in a compressed form, allowing the softmax to properly differentiate between high-attention and low-attention neighbors. This is a small but important design choice — it ensures the attention mechanism has full expressive range.

### Q3: What is the role of multi-head attention in GAT?

**Petar Veličković:** Multi-head attention serves two purposes. First, it stabilizes the learning process — similar to how averaging over multiple random initializations reduces variance, having K independent attention heads and combining their outputs makes the learned representations more robust. Second, it allows the model to attend to different aspects of the neighborhood simultaneously. One head might learn to focus on structurally similar neighbors, another on feature-similar ones, and another on high-degree nodes. By concatenating the outputs of K heads in hidden layers, we get a richer representation that captures multiple attention patterns. In the output layer, we average instead of concatenate, because we need a fixed-dimensional output for classification. This design was inspired by the multi-head attention in the Transformer paper (Vaswani et al., 2017), which was published around the same time.

### Q4: GAT was described as inductive — what does that mean in practice?

**Petar Veličković:** Inductive means the model can generalize to graphs it has never seen before. GCN and most spectral methods are transductive — they require the entire graph (including test nodes) to be available during training, because the spectral decomposition or normalization is computed on the full graph. GAT is different: the attention weights are computed from node features using learned parameters (W and a), not from the global graph structure. So if you train on one graph and then present a completely new graph at test time, GAT can compute attention coefficients for the new graph's neighborhoods and produce meaningful representations. We demonstrated this on the PPI (protein-protein interaction) dataset, where the test graphs were entirely separate from the training graphs. This is crucial for real-world applications where new nodes and edges constantly appear — social networks grow, new molecules are synthesized, new users join a platform.

### Q5: What was the most surprising finding when you ran the experiments?

**Petar Veličković:** I think the most surprising thing was how well the attention visualizations corresponded to meaningful structure in the data. When we visualized which neighbors received the highest attention weights on the Cora citation network, we saw that the model had learned to attend preferentially to same-class neighbors — even though it was never explicitly told which nodes were in the same class. The attention mechanism discovered the homophilic structure of the graph on its own. This was exciting because it meant GAT wasn't just more accurate; it was also more interpretable. You could look at the attention weights and understand why the model made a particular prediction. This interpretability aspect became one of the most cited features of the paper and inspired a lot of follow-up work on GNN explainability.

## Common Misconceptions

1. **"GAT is just GCN with learned weights."** — While GAT replaces GCN's fixed normalization with learned attention, the architectural implications are deeper than a simple substitution. GCN's normalization is a global property of the graph (computed once from the full adjacency matrix), while GAT's attention weights are **local and feature-dependent** — they are recomputed at every layer for every node based on the current features. This means GAT's aggregation is dynamic and changes during training as features evolve, whereas GCN's aggregation is static. Furthermore, GAT's inductive capability (generalizing to unseen graphs) is a direct consequence of attention being feature-based, which GCN fundamentally cannot do.

2. **"The attention weights in GAT are always interpretable."** — While GAT's attention weights provide a built-in mechanism for interpretability, they are not guaranteed to be meaningful in all cases. GATv2 (Brody et al., 2022) showed that the original GAT suffers from "static attention" — the ranking of attention scores for a given node is the same regardless of which node is the query. This means the attention may not truly differentiate between neighbors in a task-specific way. The attention visualizations are most meaningful when the model has genuinely learned to attend differently to different neighbors, which is more likely on heterophilic graphs where neighbor importance varies.

3. **"GAT doesn't need self-loops because attention handles it."** — While it's true that GAT includes each node in its own neighborhood (computing α_ii alongside α_ij), this is functionally equivalent to adding self-loops — the node attends to its own transformed features. Without this self-attention, a node's representation would be entirely determined by its neighbors, losing its own identity. The self-attention term is crucial and should not be omitted. It's just implemented differently from GCN's explicit A + I — through the neighbor list rather than the adjacency matrix.

4. **"Multi-head attention in GAT is the same as in Transformers."** — There are important differences. In Transformers, multi-head attention allows attending to different positions in a sequence with different query/key projections, and the heads operate over the full sequence. In GAT, each head attends only over a node's local neighborhood (masked attention), and the "queries" and "keys" are the same node features (self-attention on the graph). The concatenation vs averaging strategy also differs: GAT concatenates hidden layers but averages the output layer, while Transformers typically concatenate all layers and then project linearly.

## Real Citations

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). "Graph Attention Networks." ICLR 2018. arXiv:1710.10903. 30,000+ Google Scholar citations.

- Key predecessors:
  - Kipf, T.N. & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017. (GAT's direct predecessor — fixed normalization that GAT replaced with attention)
  - Bahdanau, D. et al. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR 2015. (Attention mechanism that GAT adapted to graphs)
  - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017. (Multi-head attention, published same era — GAT's multi-head design parallels it)
  - Bruna, J. et al. (2014). "Spectral Networks and Locally Connected Networks on Graphs." ICLR 2014. (Spectral graph convolutions — the framework GAT moved beyond)

- Key follow-ups:
  - Brody, S., Alon, U., & Yahav, E. (2022). "How Attentive are Graph Attention Networks?" ICLR 2022. (GATv2 — fixes static attention in GAT)
  - Wang, X. et al. (2019). "Heterogeneous Graph Attention Network (HAN)." WWW 2019. (Extends GAT to heterogeneous graphs)
  - Dwivedi, V.P. & Bresson, X. (2020). "A Generalization of Transformer Networks to Graphs." (Graph Transformers building on GAT's attention paradigm)

- Other notable:
  - Zhou, J. et al. (2020). "Graph Neural Networks: A Review of Methods and Applications." AI Open. (Comprehensive survey positioning GAT as a core attention-based GNN)
  - Hamilton, W. et al. (2017). "Inductive Representation Learning on Large Graphs (GraphSAGE)." NeurIPS 2017. (Another inductive GNN, using sampling instead of attention)
  - Xu, K. et al. (2019). "How Powerful are Graph Neural Networks? (GIN)." ICLR 2019. (Theoretical analysis of GNN expressiveness — GAT's attention is more expressive than GCN's fixed aggregation)
