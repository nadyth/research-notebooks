# Deep Reinforcement Learning from Human Preferences

**Paper:** Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). *Deep reinforcement learning from human preferences.* arXiv:1706.03741.

**arXiv:** https://arxiv.org/abs/1706.03741

**Citations:** ~499 (OpenAlex, as of 2026)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/deep-rl-from-human-preferences)

## Summary

This paper introduces a practical method for training deep reinforcement learning agents from human feedback alone, without access to the environment's true reward function. Instead of asking humans to specify a reward function (which is hard for complex tasks) or provide demonstrations (which requires expertise), the authors ask non-expert humans to compare pairs of short trajectory segments and say which one they prefer. A reward model is trained on these pairwise comparisons, and a standard RL algorithm (PPO or TRPO) uses the learned reward to train the policy. The key insight is that relative preferences are far easier for humans to provide than absolute reward values, and only a tiny fraction of the agent's experience needs human labeling — less than 1% of interactions. The authors demonstrated this on Atari games and simulated robot locomotion (MuJoCo), achieving complex novel behaviors with roughly one hour of total human time.

## Core Idea

The method has three alternating components:

1. **Policy Learning (RL):** An RL agent (using PPO or TRPO) interacts with the environment, collecting trajectories. The reward comes entirely from the learned reward model — not from the environment.

2. **Preference Elicitation:** Periodically, pairs of short trajectory segments (1–2 seconds) are sampled from the agent's recent experience. A human (or simulated oracle) compares the two segments and states which is preferred. This creates a dataset of pairwise preferences.

3. **Reward Model Training:** A neural network reward model is trained on the preference data. Given a trajectory segment, it outputs a scalar reward. The model is trained so that the probability of preferring segment A over B is `σ(r(A) − r(B))`, where `r` is the cumulative reward model output over the segment. This is the Bradley-Terry model.

The three components alternate: the policy collects new experience, new preferences are elicited on recent trajectories, and the reward model is updated. This creates a feedback loop where the reward model improves as the policy explores new regions, and the policy improves as the reward model becomes more accurate.

### Key Method Details

- **Reward model architecture:** Same network architecture as the policy (CNN for Atari, MLP for MuJoCo), with a scalar output head.
- **Segment length:** Typically 1–2 seconds of agent experience (25–50 steps).
- **Preference model:** Bradley-Terry — `P(σ_A ≻ σ_B) = 1 / (1 + exp(r(σ_B) − r(σ_A)))`, where `r(σ) = Σ_t r(s_t, a_t)`.
- **Training frequency:** Reward model is retrained periodically (e.g., every N policy updates) on all accumulated preference data.
- **Exploration:** The RL agent's natural exploration drives diversity in the segments shown to the human.
- **Feedback efficiency:** With ~700 preferences (a few hundred to ~5500 depending on task), the agent learns complex behaviors.

## What Problem Does It Solve

Imagine you're teaching a robot to do a backflip. You can't write down a math formula that says exactly what a "good backflip" looks like — there are too many moving parts. But if someone shows you two videos of the robot trying, you can easily point and say "that one looked more like a backflip." That's the key insight: **humans are great at comparing two options but terrible at writing exact rules.** This paper solves the problem of how to train AI agents for complex tasks by only asking humans simple "which is better, A or B?" questions. Instead of needing thousands of hours of expert demonstrations or a perfect mathematical reward function, you need less than an hour of a regular person's time clicking on the better video clip. This made it practical to train AI for tasks where defining the goal precisely was previously impossible — and it became the foundation for how modern language models like ChatGPT are aligned to human preferences (RLHF).

## Influence

This paper is one of the foundational works in **Reinforcement Learning from Human Feedback (RLHF)**, which later became the core technique for aligning large language models (used in InstructGPT, ChatGPT, Claude, and others). The idea of training a reward model from pairwise preferences and then optimizing it with RL directly shaped the modern AI alignment pipeline. With ~499 citations, it bridged the gap between theoretical preference learning and practical deep RL, demonstrating that human feedback could scale to state-of-the-art RL systems with minimal human oversight.
