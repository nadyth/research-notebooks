# Soft Actor-Critic (SAC): Off-Policy Maximum Entropy Deep RL with a Stochastic Actor

**arXiv:** [https://arxiv.org/abs/1801.01290](https://arxiv.org/abs/1801.01290)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/soft-actor-critic-sac)

## Summary

Soft Actor-Critic (SAC), introduced by Haarnoja et al. (2018) at Berkeley AI Research, is an off-policy actor-critic deep reinforcement learning algorithm based on the **maximum entropy reinforcement learning framework**. Unlike standard RL, which seeks only to maximize expected reward, SAC augments the objective with an entropy term: the agent must both maximize reward *and* act as randomly as possible. This dual objective yields three concrete benefits — improved exploration (the policy tries diverse behaviors before committing), robustness to model and estimation errors, and the ability to capture multiple near-optimal modes of behavior. The algorithm combines three key ingredients: (1) an off-policy actor-critic architecture with separate policy (actor) and Q-function (critic) networks, (2) a **twin Q-network** architecture (à la TD3) that takes the minimum of two independent Q-estimates to suppress Q-value overestimation, and (3) **automatic temperature tuning** — the entropy temperature α is adjusted during training to match a target entropy rather than being a fragile hyperparameter. SAC demonstrated state-of-the-art performance on a range of continuous control benchmark tasks (MuJoCo: HalfCheetah, Ant, Humanoid, etc.), outperforming both on-policy methods (PPO, A3C) and off-policy methods (DDPG, TD3), while being remarkably stable across random seeds — a property notably absent in DDPG.

### Core Idea

The maximum entropy RL objective augments the standard reward objective:

> **J(π) = Σ E[s_t,a_t ~ ρ^π] [ r(s_t, a_t) + α · H(π(·|s_t)) ]**

where **H(π)** is the entropy of the policy and **α** is the temperature controlling the reward–entropy trade-off. The optimal policy under this objective succeeds at the task while remaining as stochastic as possible.

**Key algorithmic components:**

1. **Soft Q-value update (critic):** The critic learns the soft Q-function using a Bellman backup that incorporates entropy:
   - Q(s,a) ← r(s,a) + γ · E_s'[ V(s') ], where V(s') = E_a'[ Q(s',a') − α·log π(a'|s') ]
   - Two Q-networks (Q1, Q2) are trained independently, and the *minimum* of the two is used as the target value to avoid overestimation bias.

2. **Policy update (actor):** The actor is trained to maximize the expected soft Q-value while maximizing entropy:
   - J(π) = E_s[ E_a~π [ Q(s,a) − α·log π(a|s) ] ]
   - The policy is a tanh-squashed Gaussian, enabling reparameterization: a = tanh(μ + σ·ε), ε ~ N(0,1), so gradients flow through the sampler.

3. **Automatic temperature (α) tuning:** Instead of fixing α, SAC adjusts it to satisfy a target entropy constraint (typically H_target = −|A|, the negative of the action dimension):
   - α is learned via a dual gradient descent step that pushes the policy entropy toward the target.

4. **Off-policy replay:** All experience is stored in a replay buffer and reused, giving SAC excellent sample efficiency compared to on-policy methods.

### What Problem Does It Solve

Imagine you're teaching a robot to walk. With older methods, the robot either tries exactly one strategy at a time (very slow — it needs millions of attempts) or tunes knobs so delicately that changing the floor slightly breaks everything. SAC fixes both problems at once. It's like the robot is told: *"Get to the goal — but don't just learn one rigid way of walking; always keep a few backup styles in your back pocket."* Because the robot keeps exploring multiple ways to walk (the "entropy" part), it discovers robust solutions that work even when conditions change, and because it learns from replaying past attempts (the "off-policy" part), it doesn't waste data the way on-policy methods do. The result: faster learning, more stable results, and far less hyperparameter babysitting — so the same algorithm just works on a cheetah robot, an ant, or a humanoid without special tuning.

### Influence

SAC has become one of the most widely used and recommended algorithms for continuous control in deep RL. It is the default go-to off-policy algorithm in major RL libraries (OpenAI Spinning Up, Stable-Baselines3, RLlib). Its maximum-entropy formulation influenced subsequent work in offline RL, world models, and RLHF (reinforcement learning from human feedback). The paper has been cited thousands of times and is considered a foundational modern RL algorithm alongside PPO.

### Key Method Details Relevant to the Code

- **Twin Q-networks** (Q1, Q2) with a shared or separate target network updated via Polyak averaging (soft target updates).
- **Reparameterization trick** for the Gaussian policy: actions are sampled as a = tanh(μ(s) + σ(s)·ε), enabling low-variance policy gradients.
- **Automatic temperature α** via a learned log-α optimizer with a target entropy of −|action_dim|.
- **Replay buffer** with uniform sampling and batch updates.
- The notebook implements all of the above from scratch in PyTorch on the Pendulum-v1 Gymnasium environment, with reward curve and entropy plots.

### Simplifications vs. the Full Paper

- Uses a single Gymnasium environment (Pendulum-v1) rather than the full MuJoCo suite.
- Network sizes are small (2-layer MLPs with 256 units) to keep training within Kaggle GPU time limits.
- Episodes are capped at 200 steps; training runs for ~50,000 environment steps.
- No rendering or video logging; metrics are plotted as static charts.
