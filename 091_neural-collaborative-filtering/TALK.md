# TALK.md — Neural Collaborative Filtering

## Press & Blog Coverage

1. **Google AI Blog — "Recommending What Video to Watch Next: A Multitask Ranking System"** (2019): References NCF as foundational work in neural recommendation, noting that "replacing inner product with neural networks for user-item interaction (He et al., 2017) opened new directions."
   - URL: https://ai.googleblog.com/2019/10/recommending-what-video-to-watch-next.html

2. **Facebook AI (FAIR) — "Deep Learning Recommendation Model (DLRM)"** (2019): The DLRM paper and open-source release explicitly cite NCF as prior work that motivated neural approaches to recommendation, adopting the embedding + interaction layer paradigm.
   - URL: https://arxiv.org/abs/1906.00091

3. **AWS Machine Learning Blog — "Build a recommendation system using neural collaborative filtering"**: Amazon published a tutorial-style blog post implementing NCF with PyTorch on SageMaker, citing its effectiveness for implicit feedback scenarios.
   - URL: https://aws.amazon.com/blogs/machine-learning/build-a-recommendation-system-using-neural-collaborative-filtering-on-amazon-sagemaker/

4. **Papers With Code**: NCF is listed with benchmark results on the MovieLens platform page.
   - URL: https://paperswithcode.com/paper/neural-collaborative-filtering

## Interview-Style Q&A

**Q1: Why not just use deeper matrix factorization — add more latent dimensions?**
A: Increasing latent dimensions in MF just adds more linear factors. The fundamental limitation is that dot product is a linear, bilinear interaction — it can't capture non-linear relationships. A neural network can learn arbitrary interaction functions, including non-linear ones like "user likes item only if both feature X and feature Y are present."

**Q2: Why combine GMF and MLP instead of just using the MLP?**
A: GMF preserves the strength of matrix factorization — it's very effective at capturing linear latent factor interactions, which are a good first-order approximation. MLP captures non-linear interactions. The paper's experiments showed that GMF alone already outperforms standard MF, and MLP alone outperforms GMF, but NeuMF (their combination) outperforms both, suggesting they capture complementary patterns.

**Q3: How does negative sampling work and why is it needed?**
A: In implicit feedback, we only observe positive interactions (user clicked/watched). We don't know if a non-interaction means "dislike" or "just hasn't seen it yet." Negative sampling treats unobserved interactions as negative labels. For each positive interaction, we sample 4 items the user hasn't interacted with as negatives. This is more practical than treating all unobserved as negative (which is overwhelmingly many).

**Q4: What's the role of pre-training in NeuMF?**
A: NeuMF has separate GMF and MLP branches with different embedding sizes. Random initialization makes it hard for the optimizer to find good solutions because the two branches start from different scales. Pre-training GMF and MLP separately first, then initializing NeuMF with those weights (with a scaling factor for the MLP branch), gives better convergence and final performance.

**Q5: What are the limitations of NCF?**
A: NCF is a pointwise model — it scores each (user, item) pair independently, without considering the ranking of items relative to each other. It also doesn't model sequential or temporal patterns. Later work like BPR (pairwise), sequential models (SASRec, BERT4Rec), and graph-based models (LightGCN) addressed some of these limitations.

## Common Misconceptions

1. **"NCF replaced matrix factorization entirely"** — Not exactly. GMF is a neural generalization of MF, and NeuMF includes a GMF branch. The paper frames MF as a special case of NCF, not as something to discard.

2. **"NCF needs GPU and huge data"** — The model is lightweight. With small embeddings (8-32 dims) and MovieLens-100k, it trains in minutes on CPU. The neural architecture scales well but isn't inherently expensive.

3. **"NCF is the best recommendation model"** — While groundbreaking, subsequent models (DeepFM, LightGCN, EASE) have shown competitive or better results. NCF's contribution was demonstrating that neural interaction functions work, not claiming optimality.

## Real Citations

- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). Neural Collaborative Filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW '17)*, 173–182.
- He, X., & Chua, T.-S. (2017). Neural Factorization Machines for Sparse Predictive Analytics. *SIGIR 2017*. (Related work by same first author)
- Lian, J., Zhou, X., Zhang, F., Chen, Z., Xie, X., & Sun, G. (2018). xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems. *KDD 2018*. (Cites NCF as inspiration)
- Naumov, M., et al. (2019). Deep Learning Recommendation Model for Personalization and Recommendation Systems. *arXiv:1906.00091*. (DLRM cites NCF)
- Rendle, S., et al. (2020). Neural Collaborative Filtering vs. Matrix Factorization Revisited. *RecSys 2020*. (Critical re-examination of NCF results)
