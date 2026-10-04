# Proximal Policy Optimization Algorithms (PPO)

**Paper:** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
**Authors:** John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov  
**Published:** arXiv:1707.06347 (submitted Jul 2017, revised Aug 2017)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/proximal-policy-optimization-algorithms-ppo)

## Summary

PPO (Proximal Policy Optimization) is a family of policy gradient methods for reinforcement learning that strikes a practical balance between sample efficiency, simplicity, and wall-clock time. The key insight is a **clipped surrogate objective** that enables multiple epochs of minibatch stochastic gradient ascent on the same collected trajectory data — something vanilla policy gradient methods cannot do safely because they would make destructively large policy updates.

The standard policy gradient objective L^PG = E[log π(a|s) · Â] allows only a single gradient step per data sample. TRPO constrains updates via a KL divergence trust region, but requires complex second-order optimization (conjugate gradient). PPO replaces the hard constraint with a simple **clip**: the probability ratio r(θ) = π_θ(a|s) / π_θold(a|s) is clipped to [1−ε, 1+ε], and the surrogate objective L^CLIP = E[min(r·Â, clip(r, 1−ε, 1+ε)·Â)] forms a pessimistic lower bound on the unclipped objective. This means the optimizer gets no reward for moving the policy too far from what it was when the data was collected, but still gets penalised for making it worse. The result is an algorithm almost as simple as vanilla policy gradient, nearly as stable as TRPO, and empirically better than both.

Key method details relevant to the code:
- **Clipped surrogate objective:** L^CLIP(θ) = E_t[min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t)] where r_t = π_θ/π_θold and ε = 0.2.
- **Actor-critic architecture:** a shared or separate policy network (actor) and value network (critic). The total loss combines the policy surrogate, value function error (squared), and an entropy bonus for exploration.
- **Generalized Advantage Estimation (GAE):** Â_t = δ_t + (γλ)δ_{t+1} + ... where δ_t = r_t + γV(s_{t+1}) − V(s_t), reducing variance in advantage estimates.
- **Multiple optimization epochs:** collected trajectory data is reused for K epochs of minibatch SGD (K=10 for MuJoCo, K=3 for Atari), unlike vanilla PG which does one update per sample.
- **Clipping prevents large updates:** when advantage is positive, r is capped at 1+ε; when negative, at 1−ε. This asymmetric clip creates a pessimistic bound.

## What Problem Does It Solve

Imagine you're teaching a dog a new trick. If you reward it too generously for a small improvement, it might get overexcited and completely change its behaviour in a bad way — forgetting what it already knew. But if you're too cautious, learning takes forever. This is the core problem in reinforcement learning: how do you let an AI practice and improve its strategy without it accidentally destroying its own progress by changing too much at once?

Before PPO, you had two bad choices. Either you let the AI update its strategy freely (but it would often wreck itself), or you used a complicated mathematical "trust region" constraint that was hard to implement and slow to run. PPO introduces a beautifully simple fix: just cap how much the AI is allowed to change its mind in one step. It's like telling the dog, "You can adjust your approach, but only by a little bit each time." This makes training stable, fast, and easy to code — which is why PPO became the default reinforcement learning algorithm used everywhere from game-playing AI to robot training.

## Influence

PPO is arguably the most impactful reinforcement learning algorithm ever published. It became the default RL algorithm at OpenAI and across the entire field:
- **ChatGPT's RLHF:** PPO is the core algorithm used in the reinforcement learning from human feedback stage that trained InstructGPT and ChatGPT (OpenAI, 2022). The paper "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT) explicitly uses PPO.
- **Game-playing AI:** PPO was used in OpenAI Five (Dota 2) and is the standard algorithm for Atari benchmarks in most RL libraries.
- **Robotics:** PPO is the go-to algorithm for MuJoCo continuous control tasks and real-world robot learning.
- **RL libraries:** PPO is the default algorithm in Stable-Baselines3, Ray RLlib, CleanRL, and virtually every other RL framework.
- **Citations:** The paper has accumulated over 40,000 citations, making it one of the most cited papers in AI history.
- **Simplicity over TRPO:** PPO replaced TRPO as the preferred on-policy method because it requires only first-order optimization (Adam) instead of conjugate gradient with line search, and is compatible with parameter sharing and architectures using dropout.

arXiv link: https://arxiv.org/abs/1707.06347
