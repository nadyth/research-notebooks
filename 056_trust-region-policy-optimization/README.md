# Trust Region Policy Optimization (TRPO)

**Paper:** [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)  
**Authors:** John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel  
**Published:** ICML 2015 (arXiv: 1502.05477, submitted Feb 2015)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/trust-region-policy-optimization)

## Summary

TRPO introduces an iterative policy optimization procedure with a **guaranteed monotonic improvement** property. The core idea is to maximize a surrogate objective (the expected advantage under the old policy's state distribution) subject to a **trust-region constraint** on the KL divergence between the old and new policies. This constraint prevents destructively large policy updates. The practical algorithm approximates the theoretically-justified procedure using: (1) Monte Carlo trajectory sampling (the "single path" method), (2) a conjugate gradient solver for the constrained optimization instead of inverting the full Fisher information matrix, and (3) a backtracking line search that ensures the KL constraint is satisfied at each step. The method is related to natural policy gradient methods but uses a fixed KL divergence bound rather than a fixed penalty coefficient, which empirically yields more robust and consistent improvement across tasks.

Key method details relevant to the code:
- **Surrogate objective:** L(θ) = E[ (π_θ(a|s) / π_old(a|s)) · A_old(s,a) ] — the importance-weighted advantage.
- **Trust-region constraint:** D̄_KL(π_old || π_θ) ≤ δ (typically δ = 0.01), enforced via conjugate gradient + line search.
- **Fisher-vector product:** Computed analytically (Hessian of KL) using autograd, avoiding explicit matrix construction.
- **Conjugate gradient:** Solves x = F⁻¹g iteratively (typically 10 steps) without inverting the Fisher matrix.
- **GAE advantages:** Generalized Advantage Estimation for low-variance advantage estimates with a value-function baseline.

## What Problem Does It Solve

Imagine you're teaching a robot to walk. Every time you adjust how it moves, you want it to get a little bit better — never worse. But if you change its movements too drastically all at once, it might fall over and "forget" how to do anything useful. 

TRPO solves this by putting a **speed limit** on how much the robot's strategy can change in a single learning step. Think of it like adjusting the volume knob on a stereo: instead of cranking it from 0 to 100 in one twist (which might blow out the speakers), you turn it up a little, check if it sounds good, and repeat. The "speed limit" is measured by how different the new strategy is from the old one (using something called KL divergence). As long as you stay within that limit, the theory guarantees the robot's performance will **never go backwards** — it can only improve or stay the same. This makes training much more reliable, especially for complex tasks like walking, swimming, or playing video games, where one bad update could ruin everything the robot has learned so far.

## Influence

TRPO is one of the most influential papers in deep reinforcement learning. It directly inspired **PPO (Proximal Policy Optimization)** (Schulman et al., 2017), which simplifies TRPO's constrained update into a clipped surrogate objective and has become the default RL algorithm in both research and industry (e.g., OpenAI's usage, ChatGPT's RLHF training). The trust-region concept — limiting policy changes per update — is now a standard principle in policy-gradient methods. TRPO also popularized the use of conjugate gradient methods and Fisher information in deep RL, and the GAE advantage estimator (introduced in a companion paper by the same first author) became ubiquitous.
