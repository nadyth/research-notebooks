# CODE_ARCHITECTURE.md — Playing Atari with Deep Reinforcement Learning (DQN)

## Overview

The notebook implements a Deep Q-Network (DQN) from scratch following Mnih et al. (2013). The implementation includes all core components of the original paper: a Q-network (neural network function approximator), experience replay (replay buffer with random minibatch sampling), a target network (periodically updated copy for stable targets), and ε-greedy exploration. We train on CartPole-v1 (a classic control environment with 4-dim state and 2 actions) using an MLP, which preserves all algorithmic components while being CPU-runnable. The notebook visualizes the reward curve over episodes and demonstrates the agent learning to balance the pole.

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, gymnasium (RL environments), matplotlib, numpy. Set random seeds for reproducibility. Configure device (CPU or CUDA).
- **Key detail**: Uses `gymnasium` (the maintained successor to OpenAI Gym). Only pip-installable packages (torch, gymnasium, matplotlib, numpy).

### 2. Hyperparameters
- **Purpose**: Centralize all training configuration.
- **Key values**:
  - `state_dim = 4` — CartPole observation dimension (cart position, velocity, pole angle, angular velocity)
  - `action_dim = 2` — CartPole actions (push left, push right)
  - `hidden_dim = 128` — hidden layer size for the Q-network
  - `lr = 1e-3` — learning rate for Adam optimizer
  - `gamma = 0.99` — discount factor
  - `buffer_size = 10,000` — replay buffer capacity (reduced from paper's 1M for CPU)
  - `batch_size = 64` — minibatch size for training
  - `target_update_freq = 10` — steps between target network updates (reduced from paper's 10,000)
  - `epsilon_start = 1.0` — initial exploration rate
  - `epsilon_end = 0.01` — minimum exploration rate
  - `epsilon_decay = 0.995` — per-episode decay rate
  - `num_episodes = 500` — training episodes
  - `max_steps = 500` — max steps per episode (CartPole-v1 cap)
- **Data flow**: These constants flow into network construction, replay buffer, and training loop.

### 3. Environment Setup
- **Purpose**: Create the CartPole-v1 environment and inspect observation/action spaces.
- **Key functions**: `gymnasium.make('CartPole-v1')`
- **Shapes**:
  - Observation: `(4,)` — [cart position, cart velocity, pole angle, pole angular velocity]
  - Action: discrete, 2 values (0=left, 1=right)
- **Simplification vs paper**: The paper used Atari 2600 games with 84×84×4 pixel inputs and 4–18 actions. We use CartPole with 4-dim state and 2 actions for CPU tractability.

### 4. Q-Network
- **Purpose**: Neural network that approximates Q(s, a) for all actions simultaneously.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 4)` — state vector
  - Hidden: `(batch_size, 128)` — with ReLU
  - Hidden: `(batch_size, 128)` — with ReLU
  - Output: `(batch_size, 2)` — Q-value for each action (no activation)
- **Key class**: `QNetwork(nn.Module)` with `forward(state)` returning Q-values for all actions
- **Data flow**: state → linear → ReLU → linear → ReLU → linear → Q-values (one per action)
- **Key detail**: The network outputs Q-values for ALL actions in one forward pass, so `argmax` over the output gives the greedy action. This is more efficient than passing each action separately.
- **Simplification vs paper**: The paper used a CNN (3 conv layers + 2 FC layers) for pixel inputs. We use an MLP for the 4-dim CartPole state. The Q-learning algorithm is identical.

### 5. Experience Replay Buffer
- **Purpose**: Store transitions (s, a, r, s', done) and sample random minibatches to decorrelate training updates.
- **Key class**: `ReplayBuffer` with `push(state, action, reward, next_state, done)` and `sample(batch_size)` methods
- **Implementation**: Uses `collections.deque` with maxlen for automatic eviction of old transitions
- **Data flow**: Agent collects transitions → buffer → random sampling → training minibatch
- **Key detail**: The `sample()` method returns tensors ready for the network. The `done` flag is used to zero out the target for terminal states (no future reward).

### 6. Target Network
- **Purpose**: A frozen copy of the Q-network used to compute stable training targets.
- **Mechanism**: `target_net.load_state_dict(policy_net.state_dict())` every `target_update_freq` steps
- **Key detail**: The target network's parameters θ⁻ are not updated by gradients — they are periodically hard-copied from the policy network θ. This prevents the "moving target" problem where the network bootstraps from its own rapidly changing estimates.

### 7. ε-Greedy Action Selection
- **Purpose**: Balance exploration and exploitation.
- **Mechanism**: With probability ε, select a random action; otherwise select argmax_a Q(s, a; θ)
- **Schedule**: ε starts at 1.0 (full exploration) and decays to 0.01 (mostly exploitation) over training
- **Key function**: `select_action(state, policy_net, epsilon, action_dim)`

### 8. Training Loop
- **Purpose**: The main DQN training algorithm following Algorithm 1 from the paper.
- **Per-episode flow**:
  ```
  for each episode:
    reset environment → initial state s
    for each step:
      1. Select action a via ε-greedy
      2. Execute a, observe reward r and next state s'
      3. Store (s, a, r, s', done) in replay buffer
      4. If buffer has enough samples:
         a. Sample random minibatch from buffer
         b. Compute current Q-values: Q(s, a; θ)
         c. Compute target: y = r + γ * max_a' Q(s', a'; θ⁻) * (1 - done)
         d. Loss = MSE(y, Q(s, a; θ))
         e. Backprop and update θ
      5. Every target_update_freq steps: copy θ → θ⁻
      6. Decay ε
      7. s = s'
    log episode reward
  ```
- **Tracking**: Episode rewards, epsilon values, losses logged per episode
- **Key detail**: The target computation uses the target network (θ⁻), not the policy network (θ). The `done` flag zeros the future reward term for terminal states.

### 9. Visualization: Reward Curve
- **Purpose**: Plot episode reward over training. Should show an upward trend as the agent learns to balance the pole longer.
- **Output**: Raw reward curve + smoothed (moving average) curve. CartPole-v1 max reward is 500.

### 10. Visualization: Exploration Decay
- **Purpose**: Plot ε over episodes showing the exploration-to-exploitation transition.

### 11. Visualization: Loss Curve
- **Purpose**: Plot the Q-learning loss over training steps. Should decrease as Q-values stabilize.

### 12. Evaluation
- **Purpose**: Run the trained agent with ε=0 (pure greedy) for several episodes and report average reward.
- **Key detail**: This measures the agent's learned policy without exploration noise.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `QNetwork` | nn.Module | (B, 4) | (B, 2) | 3-layer MLP, outputs Q-values per action |
| `ReplayBuffer` | Class | — | — | Stores transitions, samples random minibatches |
| `select_action` | Function | (4,) | int | ε-greedy action selection |
| `compute_loss` | Function | minibatch | scalar | MSE between target and predicted Q-values |
| `train()` | Function | — | — | Main DQN training loop |

## Data Flow Diagram

```
State s ──► ε-greedy ──► Action a ──► Environment
                                         |
                                    Reward r, Next state s'
                                         |
                                    Store (s,a,r,s',done) in Replay Buffer D
                                         |
                              ┌──────────┴──────────┐
                              |                      |
                         Sample batch            (if buffer large enough)
                              |
                    ┌─────────┴─────────┐
                    |                   |
              Policy Net θ         Target Net θ⁻
              Q(s,a; θ)            Q(s',a'; θ⁻)
                    |                   |
                    |     max_a'        |
                    |        |          |
                    |   y = r + γ·max   |
                    |        |          |
                    └─── MSE(y, Q) ─────┘
                              |
                         Backprop θ
                              |
                    Every C steps: θ⁻ = θ
```

## Deliberate Simplifications vs Full Paper

1. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of Atari 2600 (84×84×4 pixels, 4–18 actions) — CPU tractable
2. **Network**: MLP instead of CNN — appropriate for the low-dimensional CartPole state
3. **Replay buffer**: 10,000 capacity instead of 1,000,000 — sufficient for CartPole
4. **Target update frequency**: Every 10 steps instead of 10,000 — appropriate for the smaller problem scale
5. **Frame preprocessing**: None — CartPole provides a clean 4-dim state; Atari requires frame stacking, grayscaling, and resizing
6. **Episode count**: 500 episodes instead of potentially thousands — CartPole converges quickly
7. **No frame skipping**: Atari DQN uses frame skip of 4; not needed for CartPole
8. **Algorithm preserved**: Experience replay, target network, ε-greedy, Q-learning loss — all core components are faithfully implemented
