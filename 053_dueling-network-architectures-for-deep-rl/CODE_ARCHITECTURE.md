# CODE_ARCHITECTURE.md — Dueling Network Architectures for Deep Reinforcement Learning

## Overview

The notebook implements both a **standard DQN** and a **Dueling DQN** from scratch in PyTorch, trains both on CartPole-v1, and directly compares learning speed and Q-value behavior. The key architectural difference is in the Q-network: standard DQN uses a single MLP that outputs Q-values directly, while Dueling DQN splits the network into a shared feature backbone, a value stream (1 output), and an advantage stream (n_actions outputs), then combines them via Q(s,a) = V(s) + A(s,a) − mean(A(s,a)). All other components (experience replay, target network, ε-greedy, optimization) are identical. The notebook tracks episode rewards and mean Q-values during training to demonstrate that the dueling architecture learns faster and produces more stable Q-value estimates.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, gymnasium, matplotlib, numpy. Set random seeds for reproducibility.
- **Key detail**: Uses `gymnasium` (maintained successor to OpenAI Gym). Only pip-installable packages.

### 2. Install Environment Library
- **Purpose**: pip-install gymnasium and import it.
- **Key functions**: `subprocess.check_call` for pip install, `gymnasium.make`

### 3. Hyperparameters
- **Purpose**: Centralize all training configuration. Identical for both DQN and Dueling DQN to ensure fair comparison.
- **Key values**:
  - `state_dim = 4` — CartPole observation dimension
  - `action_dim = 2` — CartPole actions (left, right)
  - `hidden_dim = 128` — shared hidden layer size
  - `lr = 1e-3` — learning rate
  - `gamma = 0.99` — discount factor
  - `buffer_size = 10,000` — replay buffer capacity
  - `batch_size = 64` — minibatch size
  - `target_update_freq = 10` — episodes between target network hard updates
  - `epsilon_start = 1.0`, `epsilon_end = 0.05`, `epsilon_decay = 0.995`
  - `num_episodes = 300` — training episodes
  - `max_steps = 500` — max steps per CartPole-v1 episode

### 4. Environment Setup
- **Purpose**: Create CartPole-v1, inspect observation and action spaces, run a random episode.
- **Shapes**: Observation `(4,)`, Action: discrete 2

### 5. Standard DQN Network
- **Purpose**: Baseline Q-network that outputs Q-values directly.
- **Architecture**: MLP — `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, 2)`
- **Key class**: `DQNetwork(nn.Module)` with `forward(state)` → Q-values per action
- **Data flow**: state `(batch, 4)` → hidden `(batch, 128)` → Q-values `(batch, 2)`

### 6. Dueling DQN Network
- **Purpose**: The core contribution — split network with value and advantage streams.
- **Architecture**:
  - Shared layers: `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU
  - Value stream: `Linear(128, 1)` → V(s) `(batch, 1)`
  - Advantage stream: `Linear(128, 2)` → A(s, a) `(batch, 2)`
  - Aggregation: `Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')`
- **Key class**: `DuelingDQNetwork(nn.Module)`
- **Forward pass**:
  ```python
  features = self.shared(state)        # (batch, 128)
  value = self.value_stream(features)  # (batch, 1)
  advantage = self.adv_stream(features) # (batch, 2)
  q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
  ```
- **Key detail**: The mean-subtraction (not max) is used per the paper's recommendation for stability.

### 7. Experience Replay Buffer
- **Purpose**: Store transitions (s, a, r, s', done), sample random minibatches.
- **Key class**: `ReplayBuffer` using `collections.deque` with maxlen
- **Methods**: `push(state, action, reward, next_state, done)`, `sample(batch_size)` → tensors

### 8. ε-Greedy Action Selection
- **Purpose**: Balance exploration/exploitation.
- **Mechanism**: With prob ε → random action; else → `argmax_a Q(s, a)` using online network
- **Schedule**: ε decays from 1.0 to 0.05 per episode

### 9. Optimization Step
- **Purpose**: Standard DQN optimization — identical for both networks. The only difference is which network class is used.
- **Target**: `y = r + γ * max_a' Q(s', a'; θ⁻)` — target network selects and evaluates
- **Loss**: Huber (Smooth L1) loss between `y` and `Q(s, a; θ)`
- **Key function**: `optimize_step(policy_net, target_net, buffer, optimizer, batch_size, gamma)`

### 10. Training Loop (Both Agents)
- **Purpose**: Train standard DQN and Dueling DQN with identical hyperparameters and seeds.
- **Per-agent flow**:
  ```
  for each episode:
    reset env
    for each step:
      select action (ε-greedy)
      execute, store transition
      sample minibatch
      compute loss
      gradient step
      decay ε
    update target network every target_update_freq episodes
    track: episode reward, mean Q-value
  ```
- **Tracked metrics**: `episode_rewards`, `mean_q_values` — for both agents

### 11. Comparison Plots
- **Reward curves**: Both agents' episode rewards over training, with rolling mean smoothing
- **Q-value estimates**: Mean Q-value of sampled states per episode — dueling DQN should show more stable estimates
- **Learning speed comparison**: Episodes to reach a reward threshold (e.g., 475+ on CartPole-v1)

### 12. Evaluation: Greedy Policy
- **Purpose**: Run both trained agents with ε=0 for 10 episodes, compare average rewards.

### 13. Value and Advantage Visualization
- **Purpose**: For the dueling network, extract V(s) and A(s,a) separately for sampled states and visualize how they decompose.
- **Expected result**: In states where action choice matters little, advantage values should be near-zero (both actions are similar), and V(s) should carry the signal.

### 14. Summary
- Recap of what was built and the key finding: the dueling architecture learns faster and produces more stable Q-values with the same number of parameters.

## Key Functions/Classes

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| DQNetwork | `DQNetwork(nn.Module)` | Standard Q-network with direct Q-value output |
| DuelingDQNetwork | `DuelingDQNetwork(nn.Module)` | Split value + advantage streams aggregated into Q-values |
| ReplayBuffer | `ReplayBuffer` | Stores and randomly samples transitions |
| select_action | `select_action(...)` | ε-greedy action selection |
| optimize_step | `optimize_step(...)` | DQN optimization step (works with either network) |
| train_agent | `train_agent(...)` | Full training loop for either agent type |

## Data Flow / Shapes

```
State (4,) → shared layers (128,) → value_stream → V(s) (1,)
                                 → adv_stream   → A(s,a) (2,)
Q(s,a) = V(s) + A(s,a) - mean(A)  → Q-values (2,) → argmax → action (scalar)

Action → Environment → (reward, next_state (4,), done)
(s, a, r, s', done) → ReplayBuffer → sample batch (64, ...)
Batch:
  states (64, 4) → policy_net → current_q (64, 1)     [gather at action]
  next_states (64, 4) → target_net → max → target_q (64, 1)
  target = reward + gamma * target_q * (1 - done)
  loss = HuberLoss(current_q, target)
```

## Deliberate Simplifications vs Full Paper

1. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of Atari 2600 (84×84×4 pixels, 4–18 actions). The dueling architecture's benefit is most visible with many actions (Atari has 4–18), but the principle is clearly demonstrable with 2 actions.
2. **Network**: MLP instead of CNN. No frame stacking.
3. **Buffer size**: 10,000 instead of 1,000,000. Sufficient for CartPole.
4. **Target update**: Every 10 episodes (hard copy) instead of every 10,000 steps.
5. **Episodes**: 300 instead of millions of frames.
6. **No Double DQN combination**: The paper's best results combine dueling with Double DQN. This notebook isolates the architecture effect by comparing standard DQN vs Dueling DQN with the same target computation. A follow-up could combine dueling + Double DQN.
7. **Both agents trained with same seed**: For controlled comparison.