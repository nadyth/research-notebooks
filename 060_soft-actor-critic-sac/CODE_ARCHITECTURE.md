# Code Architecture — Soft Actor-Critic (SAC) Notebook

## Section-by-Section Notebook Breakdown

### 1. Setup & Imports
- Installs `gymnasium` (the maintained successor to OpenAI Gym).
- Imports PyTorch, NumPy, matplotlib, and Gymnasium.
- Sets random seeds for reproducibility.

### 2. Hyperparameters & Configuration
- `gamma = 0.99` — discount factor.
- `tau = 0.005` — Polyak averaging coefficient for target network soft updates.
- `lr_critic = 3e-4`, `lr_actor = 3e-4`, `lr_alpha = 3e-4` — learning rates.
- `hidden_dim = 256` — width of MLP layers.
- `buffer_size = 1_000_000` — replay buffer capacity.
- `batch_size = 256` — mini-batch size for training updates.
- `start_steps = 5_000` — random exploration before training begins.
- `total_steps = 50_000` — total environment steps.
- `update_after = 1_000` — steps before training updates begin.
- `update_every = 50` — environment steps between each training round.
- `num_updates = 50` — gradient steps per training round (50 env steps → 50 updates = 1:1 ratio).

### 3. ReplayBuffer Class
- **Data stored:** state, action, reward, next_state, done (all as NumPy arrays).
- **Key methods:**
  - `store(s, a, r, s', done)` — appends a transition; wraps around circular buffer.
  - `sample(batch_size)` — returns random batch as PyTorch tensors on the device.
- **Shape flow:** states `[batch, obs_dim]`, actions `[batch, act_dim]`, rewards `[batch, 1]`, dones `[batch, 1]`.

### 4. GaussianPolicy (Actor Network)
- **Architecture:** 2-layer MLP (obs_dim → 256 → 256) producing two heads:
  - `mu_head`: Linear(256, act_dim) — mean of the Gaussian.
  - `log_std_head`: Linear(256, act_dim) — log standard deviation (clamped to [−20, 2]).
- **Forward pass:** Given state s, outputs `mu` and `log_std`.
- **Sample method:** Uses the reparameterization trick:
  - `std = exp(log_std)`
  - `epsilon ~ Normal(0, 1)`
  - `z = mu + std * epsilon` (pre-tanh raw action)
  - `a = tanh(z)` (squashed action in [−1, 1]^n)
  - `log_prob = Normal(mu, std).log_prob(z) − sum(log(1 − tanh(z)^2))`
  - The correction term accounts for the tanh squashing (Jacobian of tanh).
- **Key detail:** log_prob is summed over action dimensions.

### 5. QNetwork (Critic) & TwinQ
- **QNetwork:** 2-layer MLP (obs_dim + act_dim → 256 → 256 → 1), maps (s, a) → scalar Q-value.
- **TwinQ:** Holds two independent QNetworks (Q1, Q2) plus their target copies (Q1_target, Q2_target).
- **Key methods:**
  - `forward(s, a)` → returns Q1(s,a) and Q2(s,a).
  - `Q1(s, a)` → returns single Q-value (used in policy loss).
  - `update_targets(tau)` → Polyak averaging: θ_target ← (1−τ)·θ_target + τ·θ.

### 6. SACAgent Class
The central class tying everything together.

#### `__init__`
- Creates the actor (GaussianPolicy), critic (TwinQ), and their optimizers (Adam).
- Initializes `log_alpha` as a learnable parameter with its own Adam optimizer.
- Sets `target_entropy = −action_dim` (negative of the action space dimension).

#### `act(obs, deterministic=False)`
- Samples or takes the mean of the policy for action selection.
- During training: stochastic sample. During evaluation: deterministic (mean).

#### `update(data)`
Performs one gradient step on critic, actor, and α:
1. **Critic update:**
   - Compute next-state actions and log_probs from the actor: `a', log_prob' = actor.sample(s')`
   - Compute target V: `V_target = min(Q1_target(s', a'), Q2_target(s', a')) − α · log_prob'`
   - Compute TD target: `y = r + γ · (1 − done) · V_target`
   - Compute current Q-values: `Q1(s, a)`, `Q2(s, a)`
   - Loss: `MSE(Q1, y) + MSE(Q2, y)` — sum of both Q-network losses.
   - Gradient step on combined critic loss.

2. **Actor update (if applicable):**
   - Sample fresh actions from the policy: `a_new, log_prob_new = actor.sample(s)`
   - Compute Q-values for the new actions: `Q1(s, a_new)` (or min(Q1, Q2)).
   - Policy loss: `loss_actor = (α · log_prob_new − Q1(s, a_new)).mean()` — maximize Q while maximizing entropy.
   - Gradient step on actor loss.

3. **Alpha (temperature) update:**
   - Alpha loss: `loss_alpha = (−log_alpha · (log_prob_new + target_entropy)).mean()`
   - This pushes log_prob toward `−target_entropy = action_dim`, maintaining the desired entropy level.
   - Gradient step on alpha loss.

4. **Target network soft update:**
   - Polyak averaging on both Q-target networks with τ = 0.005.

### 7. Training Loop
- **Phase 1 — Collect:** Run the environment with the current (or random, during start_steps) policy. Store transitions in the replay buffer. Track episode rewards.
- **Phase 2 — Train:** Every `update_every` steps, perform `num_updates` gradient steps using batches from the replay buffer.
- **Phase 3 — Evaluate:** Every 5,000 steps, run 10 evaluation episodes with the deterministic policy and log average reward.
- **Logging:** Episode rewards, evaluation rewards, critic loss, actor loss, alpha value, and policy entropy are recorded for plotting.

### 8. Visualization
- **Plot 1:** Training reward curve (per episode, with rolling average).
- **Plot 2:** Evaluation reward curve (smoothed).
- **Plot 3:** Alpha (temperature) over training steps — shows automatic tuning.
- **Plot 4:** Policy entropy over training steps — should hover near the target entropy.

## Data Flow / Shapes

```
Environment (Pendulum-v1)
  → obs: [3]  (cos θ, sin θ, angular velocity)
  → action: [1]  (torque in [−2, 2], rescaled from [−1, 1])

Actor: [3] → GaussianPolicy → mu[1], log_std[1] → tanh-squashed action[1] + log_prob[1]
Critic: [3]+[1] → Q1([1]), Q2([1])
ReplayBuffer: (s[3], a[1], r[1], s'[3], done[1]) × 1M

Training batch: [256, 3] states, [256, 1] actions, etc.
```

## Deliberate Simplifications vs. Full Paper

| Aspect | Paper | This Notebook |
|--------|-------|---------------|
| Environments | MuJoCo suite (HalfCheetah, Ant, Walker, Humanoid, etc.) | Pendulum-v1 only (no MuJoCo dependency) |
| Network size | 256–512 hidden units, 2–3 layers | 256 hidden, 2 layers |
| Total steps | 1M–3M for MuJoCo | 50K (sufficient for Pendulum) |
| Architecture extras | Optional LSTM, image observations | Pure MLP, state observations |
| Evaluation | Per-task final performance over multiple seeds | Single-seed training with periodic eval |
| Reward scaling | Specific per-environment reward scaling | Direct (Pendulum rewards are already manageable) |
