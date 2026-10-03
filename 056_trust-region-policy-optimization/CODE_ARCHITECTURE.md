# Code Architecture — TRPO Notebook

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install `gymnasium` (modern Gym API), `torch`, `numpy`, `matplotlib`.
- Import all dependencies; set random seeds for reproducibility.

### 2. Environment: CartPole-v1
- Use `Gymnasium`'s `CartPole-v1` as the training environment (discrete action space, 4-dim observation).
- Observation: [cart position, cart velocity, pole angle, pole angular velocity].
- Action: {0: push left, 1: push right}.
- Reward: +1 per timestep alive; episode ends at 500 steps or pole fall.

### 3. Policy Network (Actor)
- **Class `PolicyNetwork`:** Two-layer MLP (64 hidden units, ReLU) → softmax over 2 actions.
- Outputs `log_prob(a|s)` and entropy.
- Separate from the value network.

### 4. Value Network (Critic)
- **Class `ValueNetwork`:** Two-layer MLP (64 hidden units, ReLU) → scalar value estimate V(s).
- Used for advantage computation and as a baseline.
- Trained with MSE loss on Monte Carlo / TD(λ) returns.

### 5. Rollout Collection (`collect_trajectories`)
- Run the current policy in the environment for `batch_size` timesteps.
- Record: states, actions, rewards, log_probs (under old policy), dones, values.
- Compute returns-to-go: `R_t = Σ γ^k r_{t+k}`.
- Compute GAE advantages: `A_t = Σ (γλ)^k δ_{t+k}` where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`.
- **Data shapes:**
  - states: (N, obs_dim) — e.g., (2000, 4)
  - actions: (N,) — discrete int
  - advantages: (N,) — standardized (mean 0, std 1)
  - returns: (N,) — used for value-function fitting
  - old_log_probs: (N,) — log π_old(a|s), frozen for the update

### 6. TRPO Update — Core Algorithm

#### 6a. Surrogate Loss (`surrogate_loss`)
- `ratio = exp(log_prob_new - log_prob_old)` — importance sampling ratio π_θ/π_old.
- `surrogate = mean(ratio * advantages)` — the L_θold objective.
- This is what we maximize.

#### 6b. KL Divergence (`kl_divergence`)
- For discrete policies: `KL = mean( Σ_a π_old(a|s) * [log π_old(a|s) - log π_new(a|s)] )`.
- Computed using the old and new policy's softmax outputs.
- This is the constraint function D̄_KL.

#### 6c. Fisher-Vector Product (`fisher_vector_product`)
- Uses PyTorch autograd to compute the Hessian-vector product of the KL divergence.
- Step 1: Compute KL as a scalar; get gradient `g = ∇_θ KL`.
- Step 2: Compute `g · v` (dot product with the input vector).
- Step 3: Backprop through that scalar to get `∇_θ(g · v) = H·v` (the Fisher-vector product).
- This avoids constructing the full N×N Fisher matrix.

#### 6d. Conjugate Gradient (`conjugate_gradient`)
- Solves `F·x = g` iteratively (standard CG algorithm, ~10 iterations).
- Each iteration calls `fisher_vector_product(F, v)`.
- Returns the search direction `step_dir = F⁻¹g`.

#### 6e. Line Search (`line_search`)
- Compute full step: `params_new = params_old + step_fraction * step_dir`.
- `step_fraction = sqrt(2·δ / (step_dir · F·step_dir))` — scales to satisfy the trust region.
- Backtracking: try `params_new` at `step_fraction * {1, 0.5, 0.25, ...}` until:
  1. KL constraint satisfied: `D̄_KL(old, new) ≤ δ`
  2. Surrogate improved: `L(new) ≥ 0` (positive surrogate)
- If no step satisfies both, the policy is not updated this iteration (conservative).

#### 6f. Value Function Update (`update_value`)
- Train the critic network on (states, returns) for several epochs.
- MSE loss, Adam optimizer, lr ~1e-3.
- 5-10 epochs of gradient descent per TRPO iteration.

### 7. Training Loop
```
for iteration in range(max_iterations):
    1. Collect trajectories (batch_size timesteps)
    2. Compute GAE advantages and returns
    3. TRPO policy update:
       a. Compute policy gradient g = ∇ surrogate_loss
       b. Compute step_dir = CG(F, g)  # conjugate gradient
       c. Compute step_size via Fisher quadratic
       d. Line search: backtrack until KL ≤ δ and surrogate > 0
    4. Update value function (MSE on returns)
    5. Log reward, KL, surrogate, episode lengths
```

### 8. Visualization
- **Plot 1:** Episode reward over training iterations (with moving average).
- **Plot 2:** KL divergence per update (should stay near δ=0.01).
- **Plot 3:** Surrogate objective per update (should be positive).
- **Plot 4:** Line search step fractions (shows how often backtracking occurs).

### Key Functions/Classes Summary
| Component | Purpose |
|---|---|
| `PolicyNetwork` | Actor: MLP → softmax policy |
| `ValueNetwork` | Critic: MLP → V(s) baseline |
| `collect_trajectories()` | Rollout collection + GAE |
| `surrogate_loss()` | Importance-weighted advantage objective |
| `kl_divergence()` | Mean KL between old and new policies |
| `fisher_vector_product()` | Hessian-vector product via autograd |
| `conjugate_gradient()` | Iterative solver for F⁻¹g |
| `line_search()` | Backtracking to satisfy trust-region constraint |
| `update_value()` | Critic gradient descent |

### Deliberate Simplifications vs Full Paper
1. **Environment:** CartPole-v1 (simple, discrete) instead of MuJoCo locomotion or Atari from pixels. The algorithm is identical; only the policy architecture changes.
2. **Single-path sampling only:** We use the single-path (on-policy trajectory) method. The "vine" method (multiple rollouts from rollout states) is omitted — it requires environment state restoration and is mainly beneficial for simulation-only settings.
3. **Discrete action space:** The original paper handles both discrete and continuous. We use discrete (CartPole) for simplicity; the KL and Fisher formulas are simpler.
4. **No recurrence:** The paper mentions recurrent policies for POMDPs (Atari with frame stacking). We use a simple feedforward policy on the fully-observed CartPole state.
5. **GAE instead of raw Monte Carlo:** The original TRPO used raw discounted returns; we use GAE (λ=0.95) for lower-variance advantages, which was introduced in the companion paper by the same author and is the standard modern approach.
6. **KL constraint δ=0.01:** Same as the paper's locomotion experiments.
