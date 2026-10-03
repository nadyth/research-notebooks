# Continuous Control with Deep Reinforcement Learning (DDPG)

**Paper:** [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)  
**Authors:** Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra  
**Published:** arXiv:1509.02971 (submitted Sep 2015, ICLR 2016)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/continuous-control-deep-rl-ddpg)

## Summary

DDPG (Deep Deterministic Policy Gradient) adapts the ideas behind Deep Q-Networks (DQN) to the **continuous action** domain. DQN's core trick — experience replay and a slowly-updating target network — stabilises training of a Q-value function approximator parameterised by a deep neural network. But DQN relies on maximising over a discrete action space, which is infeasible when actions are continuous (e.g. torques on a robot joint). DPG (Deterministic Policy Gradient, Silver et al. 2014) showed that the gradient of the expected return with respect to deterministic policy parameters can be computed via the critic's gradient: ∇_θ J ≈ E[ ∇_a Q(s,a) · ∇_θ μ(s|θ) ]. DPG, however, was limited to linear function approximators. DDPG combines DPG with DQN's stabilisation tricks — replay buffer, soft target updates (Polyak averaging), and batch normalisation — and demonstrates that the same algorithm, architecture, and hyper-parameters robustly solve **more than 20 simulated physics tasks** including cartpole swing-up, dexterous manipulation, legged locomotion, and car driving, achieving performance competitive with a planning oracle that has full access to the dynamics.

Key method details relevant to the code:
- **Actor network μ(s|θ^μ):** deterministic policy mapping states to a continuous action vector, with output squashed via `tanh` to the action range.
- **Critic network Q(s,a|θ^Q):** Q-value function taking (state, action) and outputting a scalar value estimate.
- **Target networks:** soft-updated copies of actor and critic, θ_target ← τ·θ + (1−τ)·θ_target with τ ≪ 1 (typically 0.001), providing stable bootstrap targets.
- **Experience replay:** random minibatches drawn from a replay buffer to decorrelate samples and reduce variance.
- **Exploration via Ornstein-Uhlenbeck noise** (or Gaussian noise) added to the deterministic action at training time, decaying over episodes.
- **Critic loss:** MSE on TD targets: y = r + γ·Q'(s', μ'(s')) ; actor loss: −Q(s, μ(s)) averaged over the batch.

## What Problem Does It Solve

Imagine you're trying to teach a robot arm to pick up a glass of water. The robot doesn't just have two choices like "push left" or "push right" — it needs to control the exact angle and force of each joint, which can be any number in a continuous range (like turning a dial to any position, not just clicking a switch). Before this paper, AI could learn games like Atari where there are a small number of buttons to press, but it struggled with real-world tasks requiring precise, continuous movements.

DDPG solves this by combining two ideas: (1) a "judge" (critic) that learns how good a specific movement is in a given situation, and (2) an "actor" that learns to pick the exact best movement. The actor quietly watches what the judge thinks is good and slowly adjusts its movements to match. To keep things stable, both the actor and judge have "shadow copies" (target networks) that update slowly, like learning from an old textbook so the lessons don't change too fast. This lets a single algorithm learn to control everything from a cartpole to a car — all without being told the physics of the task.

## Influence

DDPG became one of the most widely-used deep RL algorithms for continuous control. It established the actor-critic-with-replay-and-target-networks paradigm that underpins most modern off-policy continuous control methods. Its direct successors include **TD3** (Fujimoto et al., 2018), which addresses Q-value overestimation with twin critics and delayed policy updates, and **SAC** (Haarnoja et al., 2018), which adds entropy maximisation. These three algorithms (DDPG, TD3, SAC) remain the standard baselines for continuous control benchmarks such as MuJoCo, DeepMind Control Suite, and Gym's robotics environments. The paper also demonstrated that deep RL could solve diverse tasks with a single set of hyper-parameters, a key insight for the field's scalability.
