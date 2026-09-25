# CODE_ARCHITECTURE.md — Deep Reinforcement Learning with Double Q-learning

## Overview

The notebook implements both **vanilla DQN** and **Double DQN** from scratch in PyTorch, trains both on CartPole-v1, and directly compares Q-value overestimation and learning performance. The key algorithmic difference is one line in the target computation: vanilla DQN uses the target network for both selecting and evaluating the best next action, while Double DQN uses the online network for selection and the target network for evaluation. All other components (experience replay, target network, ε-greedy) are identical. The notebook tracks Q-value estimates during training to demonstrate that Double DQN produces less biased (lower) Q-values than vanilla DQN.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, gymnasium, matplotlib, numpy. Set random seeds for reproducibility.
- **Key detail**: Uses `gymnasium` (maintained successor to OpenAI Gym). Only pip-installable packages.

### 2. Install Environment Library
- **Purpose**: pip-install gymnasium and import it.
- **Key functions**: `subprocess.check_call` for pip install, `gymnasium.make`

### 3. Hyperparameters
- **Purpose**: Centralize all training configuration. Identical for both DQN and Double DQN to ensure fair comparison.
- **Key values**:
  - `state_dim = 4` — CartPole observation dimension
  - `action_dim = 2` — CartPole actions (left, right)
  - `hidden_dim = 128` — hidden layer size
  - `lr = 1e-3` — learning rate
  - `gamma = 0.99` — discount factor
  - `buffer_size = 10,000` — replay buffer capacity
  - `batch_size = 64` — minibatch size
  - `target_update_freq = 10` — steps between target network hard updates (in episodes)
  - `epsilon_start = 1.0`, `epsilon_end = 0.05`, `epsilon_decay = 0.995`
  - `num_episodes = 300` — training episodes (reduced for CPU tractability)
  - `max_steps = 500` — max steps per CartPole-v1 episode
- **Data flow**: These constants flow into both DQN and Double DQN training.

### 4. Environment Setup
- **Purpose**: Create CartPole-v1, inspect spaces, run a random episode.
- **Shapes**: Observation `(4,)`, Action: discrete 2

### 5. Q-Network
- **Purpose**: Neural network approximating Q(s, a) for all actions.
- **Architecture**: MLP — `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, 2)`
- **Key class**: `QNetwork(nn.Module)` with `forward(state)` → Q-values per action
- **Data flow**: state `(batch, 4)` → hidden `(batch, 128)` → Q-values `(batch, 2)`
- **Simplification vs paper**: Paper used CNN for 84×84×4 Atari frames. We use MLP for 4-dim CartPole state. The Double Q-learning algorithm is identical.

### 6. Experience Replay Buffer
- **Purpose**: Store transitions (s, a, r, s', done), sample random minibatches.
- **Key class**: `ReplayBuffer` using `collections.deque` with maxlen
- **Methods**: `push(state, action, reward, next_state, done)`, `sample(batch_size)` → tensors
- **Data flow**: Agent collects → buffer → random sample → training minibatch

### 7. ε-Greedy Action Selection
- **Purpose**: Balance exploration/exploitation.
- **Mechanism**: With prob ε → random action; else → `argmax_a Q(s, a; θ)` using online network
- **Schedule**: ε decays from 1.0 to 0.05 per episode
- **Key function**: `select_action(state, policy_net, epsilon, action_dim)`

### 8. Vanilla DQN Optimization Step
- **Purpose**: The standard DQN target computation for comparison.
- **Target**: `y = r + γ * max_a' Q(s', a'; θ⁻)` — target network selects AND evaluates
- **Loss**: Huber (Smooth L1) loss between `y` and `Q(s, a; θ)`
- **Key function**: `optimize_dqn(policy_net, target_net, buffer, optimizer, batch_size, gamma)`
- **Key detail**: `target_q = target_net(next_states).max(dim=1)` — max over actions from target network

### 9. Double DQN Optimization Step
- **Purpose**: The Double DQN target — the core contribution of the paper.
- **Target**: `y = r + γ * Q(s', argmax_a' Q(s', a'; θ); θ⁻)`
  - Step 1: `best_actions = policy_net(next_states).argmax(dim=1)` — online network selects action
  - Step 2: `target_q = target_net(next_states).gather(1, best_actions)` — target network evaluates
- **Loss**: Same Huber loss
- **Key function**: `optimize_double_dqn(policy_net, target_net, buffer, optimizer, batch_size, gamma)`
- **Key detail**: The ONLY difference from vanilla DQN is which network picks the argmax action. Selection uses online θ, evaluation uses target θ⁻.

### 10. Training Loop (Both Agents)
- **Purpose**: Train vanilla DQN and Double DQN with identical hyperparameters and seeds for fair comparison.
- **Per-agent flow**:
  ```
  for each episode:
    reset env
    for each step:
      select action (ε-greedy)
      execute, store transition
      sample minibatch
      compute loss (DQN or Double DQN target)
      gradient step
      decay ε
    update target network every target_update_freq episodes
    track: episode reward, mean loss, mean Q-value estimate
  ```
- **Tracked metrics**: `episode_rewards`, `episode_losses`, `mean_q_values` — for both agents

### 11. Comparison Plots
- **Reward curves**: Both agents' episode rewards over training, with rolling mean smoothing
- **Q-value estimates**: Mean Q-value of sampled states per episode — Double DQN should show lower (less overestimated) Q-values
- **Overestimation metric**: Difference between estimated Q-values and actual returns

### 12. Evaluation: Greedy Policy
- **Purpose**: Run both trained agents with ε=0 for 10 episodes, compare average rewards.
- **Method**: Pure greedy action selection, no exploration noise

### 13. Q-Value Overestimation Analysis
- **Purpose**: Directly measure overestimation by comparing predicted Q-values to actual observed returns.
- **Method**: For a set of states, compute Q(s, greedy_action) for both agents, then run episodes from those states and measure actual discounted returns.
- **Expected result**: Vanilla DQN Q-values exceed actual returns (overestimation); Double DQN Q-values are closer to actual returns.

### 14. Summary
- Recap of what was built and the key finding: Double DQN reduces overestimation with a one-line change.

## Key Functions/Classes

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| QNetwork | `QNetwork(nn.Module)` | MLP that outputs Q-values for all actions |
| ReplayBuffer | `ReplayBuffer` | Stores and randomly samples transitions |
| select_action | `select_action(...)` | ε-greedy action selection |
| optimize_dqn | `optimize_dqn(...)` | Vanilla DQN optimization step |
| optimize_double_dqn | `optimize_double_dqn(...)` | Double DQN optimization step (decoupled selection/evaluation) |
| train_agent | `train_agent(...)` | Full training loop for either agent type |

## Data Flow / Shapes

```
State (4,) → QNetwork → Q-values (2,) → argmax → action (scalar)
Action → Environment → (reward, next_state (4,), done)
(s, a, r, s', done) → ReplayBuffer → sample batch (64, ...)
Batch:
  states (64, 4) → policy_net → current_q (64, 1)     [gather at action]
  next_states (64, 4) → policy_net → argmax → best_actions (64, 1)  [Double DQN selection]
  next_states (64, 4) → target_net → target_q (64, 1)  [gather at best_actions for Double DQN]
                                          or max → target_q (64, 1)  [vanilla DQN]
  target = reward + gamma * target_q * (1 - done)
  loss = HuberLoss(current_q, target)
```

## Deliberate Simplifications vs Full Paper

1. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of Atari 2600 (84×84×4 pixels, 4–18 actions). Preserves all algorithmic components while being CPU-tractable.
2. **Network**: MLP instead of CNN. No frame stacking.
3. **Buffer size**: 10,000 instead of 1,000,000. Sufficient for CartPole.
4. **Target update**: Every 10 episodes (hard copy) instead of every 10,000 steps (hard copy). Faster adaptation for small environment.
5. **Episodes**: 300 instead of millions of frames. Enough to see learning and overestimation effects.
6. **No Atari preprocessing**: No frame skipping, grayscale conversion, or reward clipping.
7. **Both agents trained with same seed**: For controlled comparison (the paper averaged over multiple seeds and games).
