# Prioritized Experience Replay

**Paper:** [Prioritized Experience Replay (Schaul et al., 2016)](https://arxiv.org/abs/1511.05952)
**Authors:** Tom Schaul, John Quan, Ioannis Antonoglou, David Silver
**Year:** 2016 (ICLR 2016)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/prioritized-experience-replay)

## Summary

Prioritized Experience Replay improves the standard experience replay mechanism used in Deep Q-Networks (DQN). Instead of sampling transitions uniformly at random from the replay buffer, it samples them proportionally to how much the agent can learn from each transition -- measured by the magnitude of the temporal-difference (TD) error. Transitions with high TD error are "surprising" and contain more learning signal, so they get replayed more often. To counteract the bias introduced by non-uniform sampling, the method applies importance-sampling weights that are annealed toward 1 over the course of training. The result: DQN with prioritized replay learns roughly twice as fast and achieves state-of-the-art performance on 41 out of 49 Atari games.

## Core Idea

The method has three key components:

1. **Priority by TD error:** Each transition stored in the replay buffer is assigned a priority p_i = |delta_i| + epsilon, where delta_i is the absolute TD error from the last time the transition was replayed. New transitions get maximal priority to ensure they are seen at least once.

2. **Stochastic prioritization:** Rather than greedily always picking the highest-priority transition (which causes overfitting and loses diversity), transitions are sampled with probability P(i) = p_i^alpha / sum_k(p_k^alpha). The exponent alpha controls how aggressive the prioritization is: alpha=0 gives uniform sampling, alpha=1 gives full proportional prioritization.

3. **Importance-sampling correction:** Non-uniform sampling introduces bias into the value estimates. To correct this, each transition's update is scaled by an importance-sampling weight w_i = (N * P(i))^(-beta), normalized by the maximum weight. Beta is annealed linearly from beta_0 to 1 over training, so the bias is tolerated early (when learning is non-stationary anyway) but fully corrected near convergence.

Two variants exist: **proportional** (priority = |delta| + epsilon, implemented with a sum-tree) and **rank-based** (priority = 1/rank, implemented with stratified sampling). Both perform similarly in practice.

## What problem does it solve

Imagine you are studying for a big exam by flipping through flashcards. You have 1000 flashcards but only a few hours to study. If you shuffle the deck and pick cards uniformly at random, you will waste a lot of time on cards you already know well, while the tricky concepts you keep getting wrong might only come up once or twice. That is what standard experience replay does -- it treats every past experience equally, regardless of whether the agent already learned from it or still finds it surprising.

Prioritized Experience Replay is like sorting the flashcards so that the ones you got wrong most recently come up more often. The "surprise" (TD error) tells the system which experiences the agent still struggles with, and those get replayed more frequently. To avoid only studying the same few cards and forgetting everything else, the method adds some randomness (so every card still has a chance of appearing) and a correction weight (so that frequently-shown cards count less per appearance, keeping the overall learning balanced).

## Key Method Details

- **Priority metric:** Absolute TD error |delta| = |R + gamma * max_a Q(s', a') - Q(s, a)|
- **Sampling probability:** P(i) = p_i^alpha / sum_k(p_k^alpha), with alpha in [0, 1]
- **IS weights:** w_i = (N * P(i))^(-beta) / max_j(w_j), with beta annealed from beta_0 to 1
- **Sum-tree data structure:** For proportional variant -- a binary tree where each leaf holds a priority and each internal node holds the sum of its children. Sampling is O(log N), updates are O(log N).
- **Hyperparameters (Atari):** alpha=0.6, beta_0=0.4 (proportional); alpha=0.7, beta_0=0.5 (rank-based)
- **Step-size:** Reduced by factor 4 compared to uniform replay (because high-error transitions produce larger gradients)
- **Combined with Double DQN** to avoid overestimation bias

## Influence

Prioritized Experience Replay is one of the most widely adopted improvements in deep RL. It is a standard component in modern RL libraries (OpenAI Baselines, Stable Baselines3, Ray RLlib) and is used in combination with most DQN-derived algorithms. The paper has been cited 5,000+ times and the sum-tree data structure has become a textbook example of efficient priority sampling. The idea of prioritizing learning by surprise/error has also influenced curriculum learning, hard negative mining, and active learning in supervised settings.