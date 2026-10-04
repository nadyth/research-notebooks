# CODE_ARCHITECTURE.md — Rainbow: Combining Improvements in Deep Reinforcement Learning

## Overview

The notebook implements a simplified **Rainbow DQN** agent that combines three of the six DQN improvements — **Double DQN**, **Dueling Networks**, and **Prioritized Experience Replay** — into a single agent, and trains it on **CartPole-v1** alongside each individual component (vanilla DQN, Double DQN, Dueling DQN, PER DQN) for direct comparison of learning curves. The three chosen components are the ones implemented in earlier notebooks (052–054) and are the most algorithmically illustrative to combine. The full six-component Rainbow additionally includes multi-step returns, distributional RL (C51), and noisy nets; these are discussed in the markdown and a simplified version of multi-step returns is included as an optional extension. The notebook is self-contained and does not depend on the earlier notebooks — all code is reimplemented here.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, gymnasium, matplotlib, numpy. Set random seeds.
- **Key detail**: Uses `gymnasium` (maintained successor to OpenAI Gym). Only pip-installable packages.

### 2. Install Dependencies
- **Purpose**: pip-install gymnasium.
- **Key functions**: `subprocess.check_call` for pip install.

### 3. Hyperparameters
- **Purpose**: Centralize all training configuration.
- **Key values**:
  - `state_dim = 4` — CartPole observation dimension
  - `action_dim = 2` — CartPole actions (left, right)
  - `hidden_dim = 128` — hidden layer size
  - `lr = 1e-3` — learning rate
  - `gamma = 0.99` — discount factor
  - `buffer_size = 20,000` — replay buffer capacity (larger for PER)
  - `batch_size = 64` — minibatch size
  - `target_update_freq = 200` — steps between target network hard updates
  - `epsilon_start = 1.0`, `epsilon_end = 0.05`, `epsilon_decay = 0.995`
  - `num_episodes = 250` — training episodes per agent
  - `max_steps = 500` — max steps per CartPole-v1 episode
  - `alpha = 0.6` — PER prioritization exponent
  - `beta_start = 0.4`, `beta_end = 1.0` — PER importance-sampling annealing
  - `n_step = 3` — multi-step return horizon (optional extension)
- **Data flow**: Constants flow into all four agent variants.

### 4. Environment Setup
- **Purpose**: Create CartPole-v1, inspect spaces.
- **Shapes**: Observation `(4,)`, Action: discrete 2

### 5. Q-Network (Vanilla)
- **Purpose**: Standard MLP Q-network for vanilla DQN and Double DQN baselines.
- **Architecture**: `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, 2)`
- **Data flow**: state `(batch, 4)` → Q-values `(batch, 2)`

### 6. Dueling Q-Network
- **Purpose**: Split value/advantage streams for Dueling DQN and Rainbow.
- **Architecture**: Shared trunk `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU; then splits into:
  - Value stream: `Linear(128, 1)` → V(s)
  - Advantage stream: `Linear(128, 2)` → A(s, a)
  - Output: `Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)`
- **Key class**: `DuelingQNetwork(nn.Module)`
- **Data flow**: state `(batch, 4)` → shared `(batch, 128)` → value `(batch, 1)` + advantage `(batch, 2)` → Q `(batch, 2)`
- **Simplification vs paper**: Paper used CNN for 84×84×4 Atari frames. We use MLP for 4-dim CartPole state.

### 7. Standard Replay Buffer
- **Purpose**: Uniform random sampling for vanilla/Double/Dueling DQN.
- **Key class**: `ReplayBuffer` using `collections.deque` with maxlen
- **Methods**: `push(state, action, reward, next_state, done)`, `sample(batch_size)` → tensors

### 8. Prioritized Replay Buffer (Sum-Tree)
- **Purpose**: Proportional prioritization by TD-error magnitude for PER DQN and Rainbow.
- **Key class**: `PrioritizedReplayBuffer` using a sum-tree for O(log n) sampling
- **Methods**: `push()`, `sample(batch_size, beta)` → (tensors, indices, weights), `update_priorities(indices, td_errors)`
- **Data flow**: transitions stored with priority = |TD-error|^α; sampled proportionally; importance-sampling weights = (N·p_i)^(-β)
- **Simplification vs paper**: Paper used rank-based prioritization option too; we implement proportional (sum-tree) which is the more common variant.

### 9. ε-Greedy Action Selection
- **Purpose**: Balance exploration/exploitation for all agents.
- **Key function**: `select_action(state, policy_net, epsilon, action_dim)`
- **Note**: Rainbow's full version uses Noisy Nets for exploration instead of ε-greedy; we use ε-greedy here for simplicity and fair comparison across agents.

### 10. Optimization Functions
- **Purpose**: Four optimization functions, one per agent variant, to make the differences explicit.
- **Key functions**:
  - `optimize_vanilla_dqn()`: target = `r + γ * max_a' Q(s', a'; θ⁻)` — target net selects AND evaluates
  - `optimize_double_dqn()`: target = `r + γ * Q(s', argmax_a' Q(s', a'; θ); θ⁻)` — online selects, target evaluates
  - `optimize_dueling_dqn()`: same as Double DQN but uses DuelingQNetwork
  - `optimize_per_dqn()`: Double DQN target + PER importance-sampling weights + priority updates
  - `optimize_rainbow()`: DuelingQNetwork + Double DQN target + PER + (optional) n-step returns
- **Loss**: Huber (Smooth L1) for non-PER agents; weighted Huber for PER/Rainbow
- **Data flow**: buffer sample → compute target → loss → backward → optimizer step → (PER only) update priorities

### 11. Training Loop
- **Purpose**: Train each agent for `num_episodes` episodes, recording rewards and evaluation scores.
- **Key function**: `train_agent(agent_type, num_episodes, ...)` → returns reward history
- **Data flow**: For each episode: reset env → select action → step → push to buffer → optimize → update target net → record reward
- **Simplification**: Paper trained for 200M frames across 57 games; we train 250 episodes on CartPole-v1.

### 12. Comparison & Visualization
- **Purpose**: Plot learning curves for all agents on the same chart.
- **Output**: Matplotlib figure with smoothed reward curves for vanilla DQN, Double DQN, Dueling DQN, PER DQN, and Rainbow, plus a summary table of mean final scores.
- **Key insight**: Rainbow should show faster learning and/or higher final scores than any individual component, demonstrating the synergistic combination.

## Key Classes & Functions Summary

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| QNetwork | `QNetwork(nn.Module)` | Standard MLP Q-network |
| DuelingQNetwork | `DuelingQNetwork(nn.Module)` | Value + advantage streams |
| ReplayBuffer | `ReplayBuffer` | Uniform sampling buffer |
| PrioritizedReplayBuffer | `PrioritizedReplayBuffer` | Sum-tree proportional sampling |
| select_action | `select_action()` | ε-greedy action selection |
| optimize_* | `optimize_vanilla/double/dueling/per/rainbow()` | Per-agent optimization steps |
| train_agent | `train_agent()` | Full training loop |
| plot_comparison | `plot_comparison()` | Learning curve visualization |

## Deliberate Simplifications vs Full Paper

1. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of 57 Atari 2600 games (84×84×4 frames, 18 actions). This makes training tractable on CPU/GPU in minutes while preserving all algorithmic differences.
2. **Network architecture**: MLP instead of CNN. The dueling split, double DQN logic, and PER sampling are identical regardless of backbone.
3. **Three of six components**: We implement Double DQN + Dueling + PER (the three from earlier notebooks) as the core Rainbow combination. Multi-step returns are included as an optional extension. Distributional RL (C51) and Noisy Nets are described in markdown but not implemented, to keep the notebook focused and runnable.
4. **Training scale**: 250 episodes instead of 200M frames. The learning curve differences are still clearly visible on CartPole.
5. **Exploration**: ε-greedy for all agents instead of Noisy Nets, for fair comparison.
6. **Prioritization**: Proportional (sum-tree) PER only; rank-based variant not implemented.
