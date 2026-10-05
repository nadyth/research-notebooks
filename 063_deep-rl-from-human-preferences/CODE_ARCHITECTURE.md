# Code Architecture — Deep RL from Human Preferences

## Overview

This notebook implements a simplified version of RLHF (Reinforcement Learning from Human Preferences) following Christiano et al. (2017). The pipeline has three alternating components:

1. **Policy training** via PPO on a toy control task (CartPole-v1)
2. **Simulated human preference elicitation** using a known oracle reward
3. **Reward model training** on pairwise trajectory comparisons

## Notebook Structure

### Section 1: Environment Setup
- Install/import dependencies (`gymnasium`, `torch`, `numpy`, `matplotlib`)
- Create the CartPole-v1 environment
- Define hyperparameters (segment length, number of preferences, PPO params)

### Section 2: PPO Agent
- **PolicyNetwork:** MLP that outputs action logits (2 actions for CartPole)
- **ValueNetwork:** MLP that outputs state value estimate
- **PPO class:** 
  - `select_action()`: sample action from policy distribution
  - `update()`: clipped surrogate loss + value loss + entropy bonus
  - `collect_trajectories()`: roll out policy in env, store (s, a, r, logprob, value)
- Data flow: state (4,) → policy → action logits (2,) → softmax → sample action
- PPO update uses GAE (Generalized Advantage Estimation) for advantage computation

### Section 3: Reward Model
- **RewardModelNetwork:** MLP that maps state → scalar reward (same input dim as policy)
- Input: state (4,), Output: scalar reward
- **RewardModel class:**
  - `predict_reward(trajectory)`: sum of per-timestep rewards over a segment
  - `train_on_preferences(preference_data)`: minimize cross-entropy loss using Bradley-Terry model
    - `P(pref_A > pref_B) = sigmoid(r(A) - r(B))` where r is summed reward over segment
    - Loss: binary cross-entropy between predicted preference probability and actual label

### Section 4: Simulated Human Oracle
- **simulate_human_preference(seg_A, seg_B, true_reward_fn):**
  - Compute true cumulative reward for each segment
  - Return 1 if seg_A preferred, 0 if seg_B preferred
  - Optional: add noise (flip label with probability ε) to simulate human inconsistency
- The true environment reward (CartPole survival bonus) is used ONLY to generate preferences, never shown to the RL agent

### Section 5: Preference Collection Loop
- During policy training, periodically sample pairs of trajectory segments
- For each pair, call the simulated human oracle
- Store (seg_A, seg_B, preference_label) in preference dataset
- Segment length: ~20 steps (CartPole episodes are short)

### Section 6: Alternating Training Loop
```
for iteration in range(N):
    1. Collect trajectories with current policy (reward from reward model)
    2. Sample segment pairs, get simulated preferences, add to dataset
    3. Retrain reward model on all preferences (few epochs of SGD)
    4. Update policy with PPO using learned reward model
    5. Log metrics: policy reward (true), reward model accuracy
```

### Section 7: Evaluation & Visualization
- Plot true reward vs. iterations (should increase)
- Plot reward model accuracy on held-out preferences
- Compare: PPO with true reward vs. PPO with learned reward model
- Show sample trajectory segments and their predicted rewards

## Key Functions/Classes

| Component | Class/Function | Purpose |
|---|---|---|
| Policy | `PolicyNetwork` | Maps state → action distribution |
| Value | `ValueNetwork` | Maps state → value estimate |
| PPO | `PPOAgent` | Policy gradient with clipped surrogate |
| Reward Model | `RewardModel` | Maps state → scalar reward, trained on preferences |
| Oracle | `simulate_human_preference()` | Generates preference labels from true reward |
| Training | `train_rlhf_loop()` | Alternates preference collection, reward training, policy training |

## Data Flow / Shapes

```
State: (4,)          # CartPole observation
  → PolicyNetwork → logits (2,) → softmax → action (1,)
  → ValueNetwork → value (1,)
  → RewardModel → reward (1,)

Trajectory segment: list of (state, action, ...) tuples, length ~20
  → RewardModel summed over segment → scalar r(σ)
  → Pair (r(σ_A), r(σ_B)) → sigmoid(r_A - r_B) → preference probability
```

## Deliberate Simplifications vs. Full Paper

1. **Toy environment:** CartPole-v1 instead of Atari/MuJoCo — keeps training under 30 min on GPU
2. **Simulated human:** Oracle uses true env reward instead of real human comparisons — allows automated validation
3. **Network architecture:** Simple MLPs instead of CNNs (no visual input)
4. **Segment length:** 20 steps (shorter than paper's 25-50, appropriate for CartPole)
5. **No asynchronous preference collection:** Paper uses parallel labeling; we do it synchronously in the training loop
6. **Fewer preferences:** ~200-500 preferences instead of 700-5500, sufficient for CartPole's simplicity
7. **Reward model retrained from scratch** each iteration on full dataset (paper uses incremental training)
8. **No uncertainty modeling:** Paper fits reward model with ensemble; we use a single model
9. **PPO instead of TRPO:** PPO is simpler and well-understood; paper used both
10. **No pretraining:** Policy starts from random initialization (paper sometimes pretrains via imitation)
