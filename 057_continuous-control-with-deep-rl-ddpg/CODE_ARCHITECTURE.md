# Code Architecture — DDPG Notebook

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install `gymnasium` (modern Gym API with classic control envs), `torch`, `numpy`, `matplotlib`.
- Import all dependencies; set random seeds for reproducibility.
- Detect device (CUDA if available, else CPU).

### 2. Environment: Pendulum-v1
- Use `Gymnasium`'s `Pendulum-v1` as the training environment (continuous action space).
- Observation (3-dim): cos(θ), sin(θ), angular velocity (dθ/dt).
- Action (1-dim, continuous): torque in [-2, 2].
- Reward: -(θ² + 0.1·dθ² + 0.001·torque²) — penalises deviation from upright and excessive torque.
- Episode length: 200 steps.
- This is a classic continuous-control task that trains quickly on CPU/GPU and requires no MuJoCo.

### 3. Replay Buffer
- **Class `ReplayBuffer`:** Fixed-capacity circular buffer storing (state, action, reward, next_state, done) transitions.
- `push(s, a, r, s', done)`: appends a transition, overwriting oldest when full.
- `sample(batch_size)`: returns random minibatch as torch tensors.
- **Data shapes per sample:**
  - state: (obs_dim,) = (3,)
  - action: (act_dim,) = (1,)
  - reward: scalar
  - next_state: (obs_dim,) = (3,)
  - done: boolean

### 4. Actor Network (Deterministic Policy)
- **Class `Actor`:** MLP mapping obs_dim → hidden → hidden → action_dim.
  - Layers: Linear(obs_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, act_dim) → tanh.
  - Output scaled by `max_action` to map [-1, 1] → [-max_action, max_action].
- `forward(state)` returns the deterministic action μ(s|θ^μ).
- Initialised with small weights (per the paper's recommendation for stable initial behaviour).

### 5. Critic Network (Q-Function)
- **Class `Critic`:** MLP mapping (obs_dim + act_dim) → hidden → hidden → 1.
  - Layers: Linear(obs_dim + act_dim, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1).
- `forward(state, action)` concatenates state and action, returns Q(s, a|θ^Q) as a scalar.
- This is the Q-value approximator used for both the critic loss and the actor's policy gradient.

### 6. Target Networks & Soft Update
- `ActorTarget` and `CriticTarget`: deep copies of the main actor and critic.
- **`soft_update(net, target_net, tau)`:** θ_target ← τ·θ + (1−τ)·θ_target with τ = 0.005.
- Target networks provide stable bootstrap targets y = r + γ·(1−done)·Q'(s', μ'(s')).
- Soft (Polyak) averaging replaces DQN's periodic hard copy, giving smoother training.

### 7. Exploration Noise: Ornstein-Uhlenbeck Process
- **Class `OUNoise`:** Temporally correlated noise process for exploration.
  - Parameters: theta=0.15, sigma=0.2, mu=0.
  - `reset()`: sets internal state to zero (called at the start of each episode).
  - `sample()`: returns noise vector of shape (act_dim,), added to the deterministic action.
- OU noise is smoother than white noise and works well for physical control tasks (momentum-like exploration).
- Noise magnitude effectively decays as the actor learns better actions; we also implement explicit decay.

### 8. DDPG Agent
- **Class `DDPGAgent`:** Encapsulates all components.
  - `__init__`: creates actor, critic, target networks, optimisers (Adam, lr_actor=1e-4, lr_critic=1e-3), replay buffer, OU noise.
  - `select_action(state, add_noise=True)`: returns actor output ± exploration noise, clipped to action bounds.
  - `train()`: samples a minibatch from the replay buffer and performs one gradient step:
    1. Compute TD target: y = r + γ·(1−done)·Q_target(s', μ_target(s'))
    2. Critic loss = MSE(Q(s,a), y); update critic.
    3. Actor loss = −mean(Q(s, μ(s))); update actor (policy gradient via the critic).
    4. Soft-update target networks.
  - Noise is decayed by multiplying `noise_scale` by `noise_decay` each episode.

### 9. Training Loop
- Run `num_episodes` episodes (typically 100–150 for Pendulum on a 30-minute Kaggle budget).
- Each episode:
  - Reset environment and OU noise.
  - For each step: select action with exploration noise, step environment, store transition, train agent.
  - Record episode reward; decay noise scale.
- Print progress every 10 episodes: average reward over last 10 episodes.
- Track per-episode rewards and per-episode average action magnitude (noise decay indicator).

### 10. Evaluation
- After training, run 10 evaluation episodes with noise disabled (deterministic policy).
- Report mean ± std of evaluation rewards.
- Compare to the known Pendulum-v1 solved threshold (≈ -200 reward for a good policy).

### 11. Plotting
- **Plot 1: Episode reward over training** — shows learning curve with a rolling average window.
- **Plot 2: Action noise magnitude over episodes** — shows exploration decay.
- Both plots use matplotlib inline; saved as outputs in the executed notebook.

## Key Data Flow / Shapes

```
state (3,) → Actor → action (1,) → clip → env.step → (next_state, reward, done)
     ↓                                      ↓
  Critic(s,a) → Q-value (1,)          ReplayBuffer.push
                                          ↓
                                    sample(batch=64)
                                    ↓
                    Critic loss: MSE(Q(s,a), r + γ·Q'(s', μ'(s')))
                    Actor loss: -mean(Q(s, μ(s)))
                                    ↓
                    soft_update targets (τ=0.005)
```

## Deliberate Simplifications vs Full Paper

1. **Single environment (Pendulum-v1)** instead of 20+ MuJoCo tasks. Pendulum is free, fast, and requires no proprietary physics engine.
2. **No batch normalisation** in the networks. The paper uses BN on low-dimensional input layers, but BN interacts poorly with replay buffer sampling and is not critical for Pendulum.
3. **Smaller hidden layers (256 units)** vs 400/300 in the paper — sufficient for a 3-dim observation space.
4. **100–150 episodes** vs the paper's millions of steps — enough to see clear learning on Pendulum.
5. **Ornstein-Uhlenbeck noise** as in the paper, but with explicit decay for faster convergence within the Kaggle time budget.
6. **No pixel-based learning** — the paper demonstrates end-to-end learning from pixels, but we use state observations only (the primary contribution is the algorithm, not the observation modality).
