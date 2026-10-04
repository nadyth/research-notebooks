# Code Architecture — PPO Implementation

This document describes the notebook structure, key functions/classes, data flow, and deliberate simplifications versus the original paper.

## Notebook Structure

### 1. Setup & Imports
Installs required packages (`gymnasium`, `torch`) and imports all modules. Uses `gymnasium` (the maintained successor to OpenAI Gym) with the `CartPole-v1` environment — a classic discrete-action control task that trains quickly on GPU and clearly demonstrates PPO's mechanics.

### 2. Hyperparameters
All PPO hyperparameters in one cell, matching the paper's conventions where applicable:
- `gamma = 0.99` (discount factor)
- `epsilon = 0.2` (clip parameter — the paper's ε)
- `epochs = 10` (K — number of optimization epochs per update, matching Table 3)
- `gae_lambda = 0.95` (GAE parameter λ from Table 3)
- `steps_per_epoch = 2048` (T — horizon, matching Table 3)
- `minibatch_size = 64` (matching Table 3)
- `lr = 3e-4` (Adam stepsize from Table 3)
- `hidden_dim = 64` (two hidden layers of 64 units, matching the paper's MLP)
- `clip_ratio = 0.2`

### 3. ActorCritic Network
A single neural network with two heads:
- **Shared trunk:** 2-layer MLP with tanh activations (64 hidden units each, matching the paper).
- **Policy head (actor):** outputs logits over discrete actions → softmax distribution π(a|s).
- **Value head (critic):** outputs a scalar V(s).
- The paper notes that for the MuJoCo experiments, parameters were NOT shared between policy and value. We use a shared trunk for simplicity, which is the common practical variant and the one used in the Atari experiments (Table 5, where architecture follows A3C with shared layers).

**Key shapes:**
- Input: `(batch, obs_dim)` — e.g., `(2048, 4)` for CartPole
- Policy logits: `(batch, n_actions)` — e.g., `(2048, 2)`
- Value: `(batch, 1)`

### 4. Memory Buffer
A `RolloutBuffer` class that collects:
- `observations`, `actions`, `log_probs` (from the old policy π_θold)
- `rewards`, `masks` (done flags)
- `values` (from the critic)
- `advantages`, `returns` (computed after collection via GAE)

**Data flow:** The buffer collects T=2048 steps from a single actor (simplified from N parallel actors in the paper). After collection, GAE advantages are computed, the buffer is shuffled into minibatches, and the surrogate loss is optimized for K=10 epochs.

### 5. GAE Computation (`compute_gae`)
Implements Generalized Advantage Estimation (Eq. 11-12):
```
δ_t = r_t + γ·V(s_{t+1})·(1−done) − V(s_t)
Â_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
```
Computed backwards from the last step. Returns are computed as `R_t = Â_t + V(s_t)` for the value function loss.

### 6. PPO Update (`ppo_update`)
The core algorithm (Algorithm 1 from the paper):
1. Compute GAE advantages and returns from the collected buffer.
2. Normalize advantages (subtract mean, divide by std) — a standard stabilisation trick not in the original paper but used in virtually all modern implementations.
3. For K=10 epochs:
   a. Shuffle the buffer and split into minibatches of size 64.
   b. For each minibatch, compute:
      - `r(θ) = exp(log π_θ(a|s) − log π_θold(a|s))` — probability ratio
      - `L^CPI = r · Â` — unclipped surrogate
      - `L^CLIP = min(r·Â, clip(r, 1−ε, 1+ε)·Â)` — clipped surrogate
      - `L^V = (V_θ(s) − R)²` — value function loss (MSE)
      - `S = −Σ π·log(π)` — entropy bonus
      - `Total = −L^CLIP + c1·L^V − c2·S` (minimised, so negated)
   c. Backpropagate and step with Adam.
4. Track clipped vs unclipped objective values for plotting.

### 7. Clipped vs Unclipped Objective Plotting
During training, we record:
- Mean of `L^CPI = E[r·Â]` (unclipped) per update
- Mean of `L^CLIP = E[min(r·Â, clip(r,1−ε,1+ε)·Â)]` (clipped) per update
- The fraction of samples where clipping was active (|r−1| > ε)

These are plotted after training to show how clipping constrains the objective — when the policy tries to move too far, L^CLIP plateaus while L^CPI keeps increasing (for positive advantages) or keeps decreasing (for negative advantages).

### 8. Training Loop
```
for epoch in range(total_epochs):
    1. Collect 2048 steps using current policy
    2. ppo_update(buffer)  # 10 epochs of minibatch SGD
    3. Log episode rewards, losses, clip statistics
    4. Clear buffer
Plot: episode rewards over time, clipped vs unclipped objectives
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Environment | MuJoCo continuous control + Atari | CartPole-v1 (discrete, simple) |
| Parallel actors | N parallel actors (8–128) | Single actor (sequential) |
| Policy distribution | Gaussian (continuous actions) | Categorical (discrete actions) |
| Parameter sharing | Separate for MuJoCo, shared for Atari | Shared (Atari-style) |
| Network architecture | 2×64 MLP with tanh | 2×64 MLP with tanh (same) |
| Adaptive KL penalty | Explored as alternative | Not implemented (clip only) |
| Total timesteps | 1M (MuJoCo), 10M (Atari) | ~50K–100K (CartPole converges fast) |
| Entropy coefficient | 0.01 (Atari) | 0.01 (same) |
| Value loss coefficient | 1.0 | 1.0 (same) |
| Advantage normalization | Not mentioned | Yes (standard practice) |
| Observation normalization | Used in Atari | Not used (CartPole obs are small) |

The simplifications make the notebook runnable in under 5 minutes on a Kaggle GPU while faithfully implementing the core algorithm — the clipped surrogate objective, GAE, multiple optimization epochs, and the actor-critic loss combination.
