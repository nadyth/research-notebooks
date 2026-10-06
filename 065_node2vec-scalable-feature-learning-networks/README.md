# node2vec: Scalable Feature Learning for Networks

**Authors:** Aditya Grover, Jure Leskovec (Stanford University)
**Published:** KDD 2016 (arXiv: July 2016)
**arXiv:** https://arxiv.org/abs/1607.00653

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/065-node2vec-scalable-feature-learning-networks)

## Summary

node2vec is a semi-supervised algorithm for learning continuous feature representations (embeddings) for nodes in a network. It defines a flexible notion of a node's network neighborhood via biased second-order random walks, then optimizes a neighborhood-preserving objective using Skip-gram with negative sampling (SGD). The key innovation is two tunable parameters — **p** (return parameter) and **q** (in-out parameter) — that smoothly interpolate between breadth-first search (BFS, capturing structural equivalence) and depth-first search (DFS, capturing homophily/community membership). This flexibility lets a single algorithm learn embeddings that reflect either or both network organization principles, generalizing prior approaches like DeepWalk (uniform random walks) and LINE (BFS-like sampling). The paper demonstrates up to 26.7% improvement on multi-label node classification and up to 12.6% on link prediction over prior state-of-the-art.

## Core Idea

The algorithm treats a network like a "document" and nodes like "words." Instead of reading words in order, it generates "sentences" of nodes via biased random walks. The bias is controlled by p and q:

- **p (return parameter):** Controls the likelihood of immediately revisiting a node. High p discourages backtracking; low p keeps the walk local.
- **q (in-out parameter):** Controls whether the walk stays close to the source (q > 1, BFS-like, structural equivalence) or explores outward (q < 1, DFS-like, homophily).

The transition probability from node v to x, given the previous node was t, uses the search bias α_pq(t,x) = 1/p if d(t,x)=0, 1 if d(t,x)=1, 1/q if d(t,x)=2. These are precomputed and sampled efficiently using alias sampling in O(1) per step. The generated walks are fed to a Skip-gram model with negative sampling, exactly as in word2vec, to learn d-dimensional node embeddings.

## Key Method Details

- **2nd-order random walks:** The next node depends on both the current node and the previous node (Markov chain of order 2).
- **Alias sampling:** Precomputed transition probabilities allow O(1) sampling per walk step.
- **Skip-gram with negative sampling:** Walk sequences are treated as sentences; nodes within a context window k are positive pairs, random nodes are negative samples.
- **Edge features:** Node embeddings can be composed into edge embeddings via binary operators (average, Hadamard product, weighted-L1, weighted-L2).
- **Scalability:** All three phases (preprocessing, walk simulation, SGD optimization) are parallelizable. Scales to millions of nodes.

## What problem does it solve

Imagine you have a big group of friends at school, and you draw lines between people who are friends with each other. Now your teacher asks: "Can you guess which students belong to the same friend group, and which students play a similar role (like being the popular connector between groups)?" This is hard because some students are in the same clique (they all hang out together), while others are bridges between different groups. Before node2vec, computers could only look at friend groups in one way — either only seeing who's right next to you, or only seeing who's far away. node2vec is smart because it can do both at the same time: by tuning two simple knobs (p and q), it can explore close-by friends (to find roles like "bridge" or "hub") or far-away friends (to find communities), or anything in between. Then it turns every student into a list of numbers (an embedding) so that students with similar roles or similar friend groups end up with similar numbers. You can then use these numbers to predict things like "will these two students become friends?" or "which club does this student belong to?"

## Influence

node2vec became one of the most widely used graph representation learning algorithms, with thousands of citations. It established the paradigm of treating graph sampling as a tunable procedure rather than a fixed strategy, influencing subsequent work in graph neural networks and graph contrastive learning. The official implementation is available at http://snap.stanford.edu/node2vec.
