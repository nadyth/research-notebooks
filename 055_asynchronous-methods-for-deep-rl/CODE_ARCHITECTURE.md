# CODE_ARCHITECTURE.md — Asynchronous Methods for Deep Reinforcement Learning (A3C)

## Overview

The notebook implements a simplified single-process version of the A3C (Asynchronous Advantage Actor-Critic) algorithm from Mnih et al. (2016). The core components are faithfully reproduced: a shared global actor-critic network, n-step return computation, the advantage actor-critic loss with entropy regularization, and periodic gradient updates. We train on CartPole-v1 (4-dim state, 2 actions) using an MLP. The notebook plots **policy entropy** alongside episode reward — the key diagnostic that shows how exploration decreases as the policy converges. This is CPU-only and self-contained.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, gymnasium, matplotlib, numpy. Set random seeds.
- **Key detail**: Uses `gymnasium` (maintained successor to OpenAI Gym). All packages pip-installable.

### 2. Hyperparameters
- **Purpose**: Centralize all configuration.
- **Key values**:
  - `state_dim = 4` — CartPole observation dimension
  - `action_dim = 2` — CartPole actions (left, right)
  - `hidden_dim = 128` — hidden layer size
  - `lr = 3e-3` — learning rate (Adam; paper used RMSProp 7e-4 for Atari, we use a slightly higher LR for faster CartPole convergence)
  - `gamma = 0.99` — discount factor
  - `t_max = 5` — n-step return horizon (paper value)
  - `entropy_coef = 0.01` — entropy regularization coefficient (paper: β = 0.01)
  - `value_coef = 0.5` — value loss coefficient (paper: c_v = 0.5)
  - `max_grad_norm = 0.5` — gradient clipping (stabilizes training)
  - `num_episodes = 500` — total training episodes
  - `num_envs = 8` — number of parallel environments (simulates A3C's parallel threads)
- **Data flow**: These constants flow into network construction, environment setup, and training loop.

### 3. Actor-Critic Network
- **Purpose**: A single network that outputs both the policy (actor) and value (critic).
- **Architecture**: Shared MLP body + two heads
  - Shared body: `(B, 4)` → `(B, 128)` ReLU → `(B, 128)` ReLU
  - Actor head: `(B, 128)` → `(B, 2)` → Softmax → action probabilities
  - Critic head: `(B, 128)` → `(B, 1)` → state value
- **Key class**: `ActorCritic(nn.Module)` with `forward(state)` returning `(log_probs, value)` or `(action_probs, value)`
- **Data flow**: state → shared layers → split → actor head (policy) + critic head (value)
- **Simplification vs paper**: The paper used a CNN for Atari pixel inputs. We use an MLP for the 4-dim CartPole state. The shared-body + two-head architecture is faithful to the paper.

### 4. Parallel Environments (Multi-Env Wrapper)
- **Purpose**: Simulate A3C's parallel threads with multiple environment instances in a single process.
- **Key class**: `MultiEnv` — manages `num_envs` gymnasium environments
- **Methods**: `reset_all()`, `step_all(actions)` — vectorized step across all envs
- **Data flow**: Each env produces independent trajectories; gradients from all envs are accumulated and applied to the shared global network.
- **Key detail**: In the original paper, each thread has its own local model copy and asynchronously applies gradients. Our simplified version processes all envs in a synchronized loop but accumulates gradients from n-step returns computed in each env independently — preserving the diversity-of-experience benefit.

### 5. n-Step Return Computation
- **Purpose**: Compute the n-step discounted return for each environment.
- **Algorithm**:
  ```
  R = 0
  for k in range(t_max):
      R += gamma^k * r_k
      if done: break
  if not done:
      R += gamma^t_max * V(s_{t_max})  # bootstrap with value
  ```
- **Key detail**: If an episode terminates within the n-step window, the return is just accumulated rewards (no bootstrap). The bootstrap term uses the critic's value estimate of the state at the end of the n-step window.
- **Shape**: Scalar per environment → `(num_envs,)` tensor.

### 6. Advantage Calculation
- **Purpose**: Compute the advantage A = R - V(s_t).
- **Formula**: `A(s_t) = n_step_return - V(s_t)` where V(s_t) is the critic's value estimate at the start of the n-step window.
- **Key detail**: The advantage measures how much better the observed return is compared to what the critic expected. Positive advantage → increase probability of taken actions; negative → decrease.

### 7. Loss Function
- **Purpose**: Combined actor-critic loss with entropy regularization.
- **Components**:
  - **Policy loss** (actor): `-mean(log_prob(a_t) * A.detach())` — the policy gradient objective
  - **Value loss** (critic): `mean((R - V(s_t))²)` — MSE between n-step return and value estimate
  - **Entropy bonus**: `mean(H(π(·|s_t)))` — Shannon entropy of the policy, encouraging exploration
  - **Total**: `L = policy_loss + value_coef * value_loss - entropy_coef * entropy`
- **Key detail**: Advantages are detached (`.detach()`) so the critic's gradient doesn't flow through the actor loss. The entropy term is subtracted (we want to maximize entropy = explore more).

### 8. Training Loop
- **Purpose**: Main A3C training loop.
- **Per-iteration flow**:
  ```
  for each update step:
    for each parallel env:
      1. Run n steps in the env, collecting (s, a, r, s', done) transitions
      2. Compute n-step return R with bootstrap
      3. Accumulate: log_prob(a), value V(s), R
    Combine across envs:
      4. Compute advantage A = R - V(s_t)
      5. Compute total loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
      6. Backprop and clip gradients (max_norm)
      7. Optimizer step
    Log metrics: episode rewards, policy entropy, value loss
  ```
- **Tracking**: Episode rewards (per env), mean policy entropy, mean value loss per update step.

### 9. Visualization: Reward Curve
- **Purpose**: Plot episode reward over training. Should show an upward trend as the agent learns to balance the pole longer.
- **Output**: Raw reward curve + smoothed (moving average). CartPole-v1 max reward is 500.

### 10. Visualization: Policy Entropy
- **Purpose**: Plot policy entropy over training. This is the key diagnostic from the paper.
- **Expected behavior**: Entropy starts high (uniform policy ≈ ln(2) ≈ 0.693 for 2 actions) and decreases as the policy becomes more confident/deterministic. If entropy drops too fast, the agent may be converging prematurely. If it stays high, the agent isn't learning.
- **Key detail**: The entropy regularization term in the loss prevents entropy from dropping to zero — it maintains a minimum exploration level. The β = 0.01 coefficient controls this balance.

### 11. Visualization: Value Loss
- **Purpose**: Plot the critic's value loss over training. Should decrease as the critic learns to estimate returns accurately.

### 12. Evaluation
- **Purpose**: Run the trained policy (stochastic, no epsilon needed) for several episodes and report average reward.
- **Key detail**: Unlike DQN (which needs ε-greedy), A3C's policy is stochastic — we sample actions from π(a|s) during evaluation. For deterministic evaluation, we can also take argmax.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `ActorCritic` | nn.Module | (B, 4) | (B, 2), (B, 1) | Shared body, actor head (softmax), critic head |
| `MultiEnv` | Class | — | — | Manages N parallel gymnasium envs |
| `compute_n_step_return` | Function | rewards, gamma, V(s_end), done | scalar | n-step return with bootstrap |
| `compute_loss` | Function | log_probs, values, returns | scalar | A2C + entropy loss |
| `train()` | Function | — | — | Main training loop |

## Data Flow Diagram

```
                    ┌─── Env 0 ── n steps ── (s, a, r, s', done) ──┐
                    ├─── Env 1 ── n steps ── (s, a, r, s', done) ──┤
Shared Global Net ──┼─── Env 2 ── n steps ── (s, a, r, s', done) ──┤── Accumulate
   (θ, θ_v)        │   ...                                       │   gradients
                    └─── Env N ── n steps ── (s, a, r, s', done) ──┘
                                                                         │
                              ┌──────────────────────────────────────────┘
                              │
                    Compute n-step returns: R_i for each env
                    Compute advantages: A_i = R_i - V(s_i)
                              │
                    ┌─────────┴───────────────────┐
                    │                             │
              Actor Loss                    Critic Loss         Entropy
              -log π(a_i)·A_i              (R_i - V_i)²        H(π(·|s_i))
                    │                             │               │
                    └─────── L_total ─────────────┴───────────────┘
                              │
                         Backprop + clip
                              │
                    Optimizer step (shared θ, θ_v)
                              │
                    Sync metrics: reward, entropy, loss
```

## Deliberate Simplifications vs Full Paper

1. **Single-process synchronous** instead of multi-threaded asynchronous — the original uses Python threading with shared memory. Our version runs multiple envs in a synchronized loop within a single process. The gradient accumulation from diverse parallel experience is preserved; only the asynchronicity of gradient application is simplified.
2. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of Atari 2600 (84×84×4 pixels, 4–18 actions) — CPU tractable
3. **Network**: MLP instead of CNN — appropriate for low-dimensional CartPole state
4. **Optimizer**: Adam instead of RMSProp — Adam is more robust to hyperparameter choices and works well for small-scale RL. The paper used RMSProp with custom decay.
5. **Thread count**: 8 parallel envs instead of 16 threads — sufficient for CartPole
6. **No frame preprocessing**: CartPole provides clean 4-dim state; Atari requires frame stacking, grayscaling, resizing
7. **Algorithm preserved**: n-step returns, advantage actor-critic loss, entropy regularization, gradient clipping, parallel exploration diversity — all core A3C components are faithfully implemented
