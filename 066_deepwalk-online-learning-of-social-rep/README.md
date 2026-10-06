# DeepWalk: Online Learning of Social Representations

**Paper:** Bryan Perozzi, Rami Al-Rfou, Steven Skiena. *DeepWalk: Online Learning of Social Representations*. KDD 2014. [arXiv:1403.6652](https://arxiv.org/abs/1403.6652)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deepwalk-online-learning-of-social-rep)

## Summary

DeepWalk is the first method to apply **language modeling** (specifically, the Skip-gram model from word2vec) to **networks/graphs** for learning node representations. The key insight is elegant: if you perform truncated random walks on a graph, the sequence of nodes visited forms a "sentence" where each node is a "word." By training a Skip-gram model on these walk sequences, you learn a low-dimensional embedding for each node that captures its structural role and community membership — without requiring any node features or labels.

The algorithm has two stages: (1) **random walk generation** — for each node, generate γ short random walks of length t by repeatedly moving to a uniformly-random neighbor; (2) **representation learning** — feed all walk sequences into a Skip-gram model with a sliding context window, using stochastic gradient descent with negative sampling to update node embeddings. The resulting embeddings encode social relations (who is connected to whom, which nodes belong to the same community) in a continuous vector space that can be directly used by downstream classifiers.

### Core Idea

DeepWalk bridges two worlds: **word2vec** (which learns word embeddings from sentences by predicting context words from a center word) and **graph representation learning** (which wants vector features for nodes that reflect graph structure). The bridge is the **truncated random walk**: a short walk on a graph visits a sequence of nodes, and nodes that appear near each other in the walk tend to share communities or structural roles — exactly the same locality property that makes word2vec work on sentences. By treating walks as sentences and nodes as words, DeepWalk reuses the entire word2vec machinery (Skip-gram, negative sampling, hierarchical softmax) to produce node embeddings that are **learned from local structure only** — no global graph statistics, no matrix factorization, no labels.

### Key Method Details

- **Truncated random walks:** For each node v, generate γ random walks of length t. At each step, move to a uniformly-random neighbor. Walks are independent and can be generated in parallel. The paper uses γ=80, t=40 as defaults (we use smaller values for speed).
- **Skip-gram with context window:** For each node in a walk, predict the nodes within a window of size w. The objective maximizes log P(context | center) = log σ(v_c · v_w) for positive pairs, plus negative sampling terms.
- **Negative sampling:** Instead of normalizing over all nodes (expensive), sample k negative nodes from a unigram distribution raised to the 3/4 power (same as word2vec). This makes training O(1) per positive pair.
- **Online learning:** New walks can be generated and added incrementally; the model does not need to reprocess the entire graph. This is crucial for streaming/dynamic graphs.
- **Parallelizable:** Random walk generation and Skip-gram training are both embarrassingly parallel — no locking required during walk generation, and Skip-gram uses Hogwild-style asynchronous updates.
- **Comparison to node2vec:** DeepWalk uses **uniform** random walks (all neighbors equally likely). node2vec (Grover & Leskovec, 2016) generalizes this to **biased second-order random walks** with parameters p (return) and q (in-out), allowing interpolation between BFS (structural equivalence) and DFS (homophily) strategies. DeepWalk is the special case where p=q=1.

### What problem does it solve?

Imagine you're new to a school and want to figure out who hangs out with whom. You could try to look at the entire social network at once — but that's overwhelming for a big school with hundreds of students. 

DeepWalk's trick is like sending a bunch of robots to randomly walk around the school, each wandering from person to person, visiting friends of friends. After many short walks, the robots notice that certain groups of people keep showing up together in the same walks — the soccer team, the band kids, the gamers. DeepWalk takes all these walk notes and uses a trick borrowed from how we learn word meanings (if words appear near each other in sentences, they're probably related) to give each person a short "profile number." People who hang out together end up with similar numbers — even though the robots never looked at the whole school map at once.

This is useful because real social networks (Facebook friends, YouTube subscriptions, corporate email networks) are huge and constantly changing. DeepWalk can learn useful profiles from just small local strolls, add new walks as the network grows, and run many robots in parallel — no need to freeze the whole network and analyze it all at once.

### Influence

With over 7,000+ citations, DeepWalk is a foundational paper in **graph representation learning** and **network embedding**. It was the first to show that NLP language modeling techniques could be directly applied to graph-structured data, opening the door to node2vec (2016), GraphSAGE (2017), GAT (2018), and the broader field of graph neural networks. The paper demonstrated that DeepWalk could achieve F1 scores up to 10% higher than baselines using only 60% of the training data, and could handle graphs with millions of nodes. The idea of "random walks as sentences" became a paradigm — it influenced not only graph embedding but also sequence-based approaches to hyperbolic embeddings, knowledge graph embeddings, and network anomaly detection.
