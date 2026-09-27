# Code Architecture — Prioritized Experience Replay Notebook

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import `numpy`, `matplotlib`, `collections.deque`, `random`, `math`.
- Set random seeds for reproducibility.
- No external RL library needed -- everything is built from scratch.

### 2. Sum-Tree Data Structure
The core data structure for proportional prioritization. A binary tree where:
- Leaf nodes (at the bottom level) store priorities p_i.
- Internal nodes store the sum of their children's priorities.
- The root stores the total sum of all priorities.

```
SumTree(capacity=N)
  tree: array of size 2*N - 1  (0-indexed, root at 0)
  data: array of size N (stores transitions)
  write_ptr: cycles 0 -> N-1 -> 0 (overwrites oldest)

Operations:
  update(idx, priority):
    # Set leaf priority, propagate up to root
    tree[idx] = priority
    while idx != 0:
      idx = (idx-1) // 2
      tree[idx] = tree[2*idx+1] + tree[2*idx+2]

  get_leaf(value):
    # Traverse from root to leaf following the value
    idx = 0
    while idx < N-1:  # not a leaf
      left = 2*idx + 1
      if value <= tree[left]:
        idx = left
      else:
        value -= tree[left]
        idx = left + 1
    return (leaf_idx, tree[leaf_idx])
```

Key properties:
- `total()` = tree[0] (root sum) -- O(1)
- `update()` -- O(log N)
- `get_leaf(s)` -- O(log N), samples a leaf whose cumulative priority range contains s

### 3. Prioritized Replay Buffer
Wraps the SumTree with the PER logic:

```
PrioritizedReplayBuffer(capacity, alpha, beta, beta_increment)
  tree = SumTree(capacity)
  alpha: prioritization exponent (0=uniform, 1=full)
  beta: IS correction exponent (0=no correction, 1=full)
  epsilon: small constant to avoid zero priority (1e-6)

  add(state, action, reward, next_state, done):
    # New transitions get max priority
    max_p = max(tree priorities) or 1.0
    tree.add(transition, max_p^alpha)

  sample(batch_size):
    # Divide [0, total] into batch_size segments
    segment = total / batch_size
    for each segment:
      s = uniform(segment_start, segment_end)
      leaf_idx, priority = tree.get_leaf(s)
      P(i) = priority / total
      IS_weight = (N * P(i))^(-beta) / max_weight
    return (transitions, indices, IS_weights)

  update_priorities(indices, td_errors):
    for each (idx, td_error):
      priority = (|td_error| + epsilon)^alpha
      tree.update(idx, priority)
```

### 4. Uniform Replay Buffer (Baseline)
A simple FIFO buffer with uniform random sampling -- the standard DQN replay.
- `add(transition)`: append, overwrite oldest if full.
- `sample(batch_size)`: random.choice without replacement.
- No priorities, no IS weights.

### 5. Q-Network (Small MLP)
A simple feedforward network for the toy environment:
```
QNetwork(state_dim, action_dim):
  Linear(state_dim, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, action_dim)
```
Two instances: online network and target network (synced periodically).

### 6. Toy Environment: CartPole-like Gridworld
A simple gridworld or CartPole-v1 (via gym) where:
- State: 4-dimensional (position, velocity, angle, angular velocity) or grid position.
- Actions: 2 (left/right).
- Reward: +1 per step alive, or based on reaching goal.
- Episode terminates on failure or max steps.

For pure-from-scratch (no gym dependency): a simple 1D gridworld with a goal state, sparse reward.

### 7. DQN Agent with PER
```
DQNAgent(state_dim, action_dim, lr, gamma, epsilon, target_update_freq,
         buffer_type="prioritized", alpha=0.6, beta=0.4):
  online_net = QNetwork(...)
  target_net = QNetwork(...)  # copy of online_net
  buffer = PrioritizedReplayBuffer(...) or UniformReplayBuffer(...)

  act(state): epsilon-greedy
  learn(batch):
    states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)
    q_values = online_net(states).gather(actions)
    next_q = target_net(next_states).max()
    td_errors = rewards + gamma * next_q * (1-dones) - q_values
    loss = (weights * td_errors^2).mean()  # weighted MSE
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    buffer.update_priorities(indices, td_errors.detach())
    buffer.anneal_beta()
  update_target(): target_net.load_state_dict(online_net.state_dict())
```

### 8. Training Loop
```
for episode in range(num_episodes):
    state = env.reset()
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.buffer.add(state, action, reward, next_state, done)
        if len(buffer) > batch_size:
            agent.learn(batch_size)
        if step % target_update_freq == 0:
            agent.update_target()
        state = next_state
        if done: break
    # Log episode reward
    # Anneal beta
```

### 9. Comparison Experiment
Run two agents side by side:
- DQN + Uniform Replay
- DQN + Prioritized Replay (proportional, alpha=0.6, beta annealed 0.4->1)

Plot:
- Episode reward over training (smoothed) for both agents
- Average TD error in the buffer over time (shows how PER focuses on high-error transitions)
- Distribution of sampled priorities (histogram at different training stages)
- Sample efficiency: episodes-to-threshold for both agents

### 10. Visualization
- Reward curves (uniform vs prioritized) on same plot
- TD error distribution histogram at episode 0, 50, 100, 200
- IS weight distribution over training
- Buffer priority distribution (how priorities spread out as learning progresses)

## Data Flow / Tensor Shapes

| Stage | Shape (batch=32, state_dim=4) |
|---|---|
| state / next_state | (32, 4) |
| action | (32, 1) |
| reward / done | (32, 1) |
| q_values | (32, 1) |
| next_q (target) | (32, 1) |
| td_errors | (32, 1) |
| IS weights | (32, 1) |
| loss | scalar |

## Deliberate Simplifications vs Full Paper

1. **Small MLP** instead of a deep CNN (no Atari frames, no convolutional layers).
2. **Toy environment** (CartPole or simple gridworld) instead of 57 Atari games.
3. **Proportional variant only** (rank-based omitted for simplicity; both perform similarly).
4. **Smaller buffer** (10,000 instead of 1,000,000 transitions).
5. **Fewer episodes** (200-500 instead of 200 million frames).
6. **No reward/TD-error clipping** (not needed in the toy environment).
7. **No Double DQN** (though the paper combines PER with Double DQN; we isolate PER's effect by comparing against uniform-replay DQN with the same architecture).