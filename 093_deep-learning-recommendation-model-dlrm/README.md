# Deep Learning Recommendation Model (DLRM)

**Paper:** [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/abs/1906.00091)  
**Authors:** Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang, Narayanan Sundaraman, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu, et al. (Facebook)  
**Year:** 2019  

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deep-learning-recommendation-model-dlrm)

## Summary

DLRM is Facebook's production-grade recommendation model that unifies two traditions in recommender systems: collaborative filtering (matrix factorization with latent factors) and predictive analytics (deep networks with embeddings). The model processes categorical features through embedding tables (one per feature, all producing vectors of the same dimension) and continuous features through a "bottom MLP" that outputs a vector of matching dimension. It then computes explicit second-order interactions between all pairs of embedding/dense vectors via dot products (inspired by Factorization Machines), concatenates these interactions with the processed dense vector, and feeds the result through a "top MLP" followed by a sigmoid to produce a click probability. The paper also describes a hybrid parallelization scheme — model parallelism for the giant embedding tables (which don't fit on one GPU) and data parallelism for the MLPs — that became a benchmark for system co-design (MLPerf).

## Core Idea

The key architectural decisions in DLRM are:

1. **Embedding tables for categorical features:** Each categorical feature (e.g., user ID, location, device type) gets its own embedding table mapping integer IDs to dense vectors of dimension `d`. This generalizes the latent-factor concept from matrix factorization.
2. **Bottom MLP for continuous features:** Dense/continuous features (e.g., age, price) are passed through a fully-connected MLP that outputs a vector of the same dimension `d` as the embeddings, so they can be interacted on equal footing.
3. **Explicit second-order interactions:** Following Factorization Machines, DLRM computes dot products between all pairs of embedding vectors (and the processed dense vector). Only pairwise (second-order) interactions are computed — not higher-order — which keeps the model compact.
4. **Top MLP + sigmoid:** The dot-product interaction outputs (upper triangular, excluding self-interactions) are concatenated with the processed dense vector and fed through a final MLP, then a sigmoid produces a probability.

This is distinct from Wide & Deep (which concatenates embeddings without explicit interactions) and from NCF (which uses MLP-based interactions instead of dot products). DLRM's structured dot-product interactions mirror Factorization Machines while adding the MLP processing power of deep learning.

## What problem does it solve?

Imagine you run a shop and want to predict which product a customer will click on. You know lots of things about each customer — their age, location, what device they're using, what categories they've browsed before — and lots of things about each product. The old approaches had a problem: they could either use the categorical info (like "user is from Mumbai") OR the numerical info (like "user is 25 years old"), but they couldn't easily figure out how these things *combine*. Like, maybe "25-year-olds from Mumbai who use iPhones" click on tech gadgets much more than any of those factors alone would suggest.

DLRM solves this by turning every piece of information — whether it's a category like "Mumbai" or a number like "25" — into the same kind of mathematical object (a vector). Then it checks how every pair of these vectors interacts with each other (like checking "does age × location matter?"). Finally, a smart neural network looks at all these interactions together and predicts whether the person will click. It's like having a shop assistant who doesn't just look at each fact about you separately, but actually considers how your age, location, and browsing history *combine* to predict what you'll want to buy.

## Influence

DLRM became the industry-standard benchmark for recommendation systems. It was adopted by **MLPerf** as the official recommendation benchmark for measuring AI hardware performance. Facebook (Meta) open-sourced the implementation in both PyTorch and Caffe2, and it has been widely used for system co-design research — studying how hardware (GPUs, memory bandwidth) interacts with recommendation model architectures. The model's hybrid parallelism scheme (model-parallel embeddings + data-parallel MLPs) influenced distributed training frameworks. DLRM is cited by hundreds of subsequent papers on recommendation systems, efficient embeddings, and systems for large-scale ML.

## Key Method Details

- **Embedding tables:** Each categorical feature has a table of size `m_i × d` where `m_i` is the vocabulary size and `d` is the embedding dimension (shared across all features).
- **Bottom MLP:** Processes dense features through FC layers with ReLU, outputting a `d`-dimensional vector.
- **Interaction layer:** Computes dot products between all pairs of the processed vectors (embeddings + dense output). Only the upper-triangular pairs (i < j) are kept, yielding `n*(n-1)/2` interaction values where `n` is the total number of feature vectors.
- **Top MLP:** Concatenates the interaction outputs with the processed dense vector and passes through FC layers with ReLU, ending with a sigmoid.
- **Loss:** Binary cross-entropy (BCE) for click/no-click prediction.
- **Data:** The paper uses the Criteo Ad Click Prediction dataset (a standard CTR benchmark). Our implementation uses MovieLens-100k adapted for CTR-style binary classification.
- **Comparison with NCF:** DLRM differs from NCF (which uses MLP-based interactions between only user and item embeddings) by using dot-product interactions across *all* feature embeddings and processing continuous features through a dedicated MLP.

## arXiv Link

https://arxiv.org/abs/1906.00091
