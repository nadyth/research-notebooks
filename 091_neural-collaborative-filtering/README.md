# Neural Collaborative Filtering

**Paper:** [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)  
**Authors:** Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, Tat-Seng Chua  
**Venue:** WWW 2017 (The Web Conference 2017)  
**Year:** 2017  

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/neural-collaborative-filtering)

## Summary

Neural Collaborative Filtering (NCF) replaces the traditional matrix-factorization inner product between user and item latent vectors with a learned neural network. Instead of computing a simple dot product to estimate how much a user likes an item, NCF feeds user and item embedding vectors into a multi-layer perceptron (MLP) that can learn arbitrary, non-linear interaction patterns from data. The paper also introduces a generalized matrix factorization (GMF) layer — a neural version of the classic dot product — and combines it with the MLP via a final fusion layer to create the NeuMF model, which captures both linear and non-linear user-item interactions. Experiments on Movielens and Pinterest datasets show NCF significantly outperforms state-of-the-art matrix factorization methods.

## Core Idea

The key insight is that the dot product in matrix factorization is a **fixed, linear** interaction function. While effective, it limits the model's ability to capture complex user-item relationships. NCF generalizes this by:

1. **GMF Layer:** User and item embeddings are element-wise multiplied — a neural generalization of matrix factorization that still allows the model to learn separate latent factors.
2. **MLP Layer:** User and item embeddings are concatenated and passed through stacked fully-connected layers with ReLU activations, learning non-linear interactions.
3. **NeuMF Fusion:** The outputs of GMF and MLP are concatenated and passed through a final sigmoid layer, combining the strengths of both linear and non-linear modeling.

Training uses **implicit feedback** (binary: interacted or not) with a pointwise loss (BCE with negative sampling), rather than explicit ratings. The model is trained on positive interactions plus sampled negative items.

## What problem does it solve?

Imagine you have a friend who always watches action movies on Netflix. The old way Netflix guessed what you'd like was simple: it found people who watched similar movies as you and recommended what they watched. But this is like a very basic matching game — it only looks at overlaps. The problem is that your taste isn't just a simple overlap. Maybe you love action movies **and** comedies, but only when they have a certain style. The old method (matrix factorization) is like a friend who only checks "do you both like the same movies?" with a simple yes/no dot product. It can't capture complicated patterns like "you like action movies but only funny ones."

NCF fixes this by replacing that simple yes/no check with a **smart neural network** that learns complex patterns. It's like upgrading from a friend who just counts movie overlaps to a friend who actually understands **why** you like certain combinations of things. This smart friend (the neural network) can figure out the tricky, non-obvious patterns in your taste that the simple counting method completely misses.

## Influence

NCF became one of the most cited papers in recommender systems, with thousands of citations. It opened the door for deep learning in collaborative filtering, inspiring architectures like DeepFM, xDeepFM, and the Deep Learning Recommendation Model (DLRM). The official implementation by the authors (the `NeuMF.py` / `NCF.py` codebase on GitHub) became a reference implementation for neural recommendation research.

## Key Method Details

- **Embeddings:** Separate embedding tables for users and items. GMF and MLP use independent embedding tables (different sizes).
- **Negative sampling:** For each positive (user, item) interaction, sample 4 negative items the user hasn't interacted with.
- **Loss:** Binary cross-entropy (BCE) treating recommendation as a binary classification problem.
- **Pre-training:** NeuMF benefits from pre-training the GMF and MLP models separately, then initializing the fused model with those weights.
- **Evaluation:** Leave-one-out evaluation — the last interacted item is held out for each user; Hit Ratio@10 and NDCG@10 are reported.

## arXiv Link

https://arxiv.org/abs/1708.05031
