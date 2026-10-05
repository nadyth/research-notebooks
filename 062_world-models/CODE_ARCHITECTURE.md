# Code Architecture — World Models

This document describes the notebook structure, key functions/classes, data flow, and deliberate simplifications versus the original paper.

## Notebook Structure

### 1. Setup & Imports
Installs `torch`, `numpy`, `matplotlib`. All modules implemented from scratch — no external RL framework or environment library beyond numpy/gym-free toy env.

### 2. Toy Environment: PointMassNav
A minimal 2D navigation environment (instead of CarRacing-v0):
- **State:** `(x, y)` position of a point-mass in a [-1, 1]² arena
- **Action:** 2D continuous velocity vector `(dx, dy)` (clipped)
- **Observation:** 64×64 grayscale image with the agent rendered as a bright dot + a target rendered as a cross
- **Reward:** −distance to target at each step; +10 on reaching target
- **Episode length:** 100 steps

**Why PointMassNav?** CarRacing-v0 requires Box2D + rendering, is GPU-heavy, and episodes take hundreds of steps. PointMassNav captures the essence — a visual observation, continuous control, sequential dynamics — in a lightweight package that trains in minutes on CPU.

### 3. VAE Module ("V")
A convolutional variational autoencoder:
- **Encoder:** 64×64×1 → conv(32,4,2,1) → conv(64,4,2,1) → conv(128,4,2,1) → flatten → FC(256) → μ(32), logσ²(32)
- **Reparameterization:** z = μ + σ ⊙ ε, where ε ~ N(0, I)
- **Decoder:** z(32) → FC(8×8×128) → reshape → deconv(128,4,2,1) → deconv(64,4,2,1) → deconv(1,4,2,1) → 64×64×1
- **Loss:** BCE reconstruction + KL divergence (β=1.0)

**Key shapes:**
- Input frame: `(B, 1, 64, 64)`
- Latent z: `(B, 32)`
- Reconstructed frame: `(B, 1, 64, 64)`

### 4. MDN-RNN Module ("M")
An LSTM with a mixture-density-network output head:
- **Input:** `[z_t (32), a_t (2)]` concatenated → 34 dims
- **LSTM:** 1 layer, hidden_size=128, processes a sequence of (z, a) pairs
- **MDN head:** FC from hidden(128) → (K × (2 mean + 2 log_std + 1 mixture_weight)) where K=5 mixtures, output dim = K × 5 = 25
- **MDN loss:** negative log-likelihood of z_{t+1} under the predicted mixture

**Data flow:**
1. Collect random rollouts in the real env
2. Encode all frames through trained VAE → z sequences
3. Train MDN-RNN to predict z_{t+1} from (z_t, a_t, hidden_t)

### 5. Controller ("C")
A linear policy:
- **Input:** `[z_t (32), h_t (128)]` → 160 dims
- **Output:** `a_t (2)` — continuous action
- **Parameters:** 160 × 2 + 2 = 322 weights (a single linear layer, no hidden layer)
- **Activation:** tanh (to bound actions to [-1, 1])

### 6. Dream Environment
A `DreamEnv` class that replaces the real simulator:
- **`reset():** Sample initial z_0 from the VAE prior (or from a random real frame), reset LSTM hidden state h_0
- **`step(action):** Feed (z_t, a_t) to MDN-RNN → get next z distribution → sample z_{t+1}, update h_{t+1}. Reward = −||decoded_z − target||² (or proxy reward from latent space).
- The controller never sees real pixels during dream training.

### 7. Training Loop (CMA-ES on Controller)
- Use a simple evolutionary strategy (CMA-ES via `cma` package, or a basic random-search evolution if cma unavailable):
  1. Sample N controller parameter vectors from a Gaussian
  2. Evaluate each in the DreamEnv for M episodes
  3. Rank by cumulative reward, update mean/covariance
  4. Repeat for ~20 generations
- **Result:** The trained controller is then tested in the **real** PointMassNav environment to measure transfer.

### 8. Evaluation & Visualization
- Run the trained controller in both the dream and real environment
- Plot: (a) VAE reconstructions (original vs decoded frames), (b) MDN-RNN prediction error over time, (c) controller reward curves (dream vs real), (d) sample trajectories in the real env
- Print final metrics: VAE loss, MDN NLL, average real-env reward

## Data Flow Summary

```
Real Env → collect random rollouts → frames
                ↓
          VAE Encoder → z sequences + actions
                ↓
          MDN-RNN trained on (z_t, a_t → z_{t+1})
                ↓
          DreamEnv = [VAE prior + MDN-RNN dynamics]
                ↓
          Controller trained in DreamEnv (CMA-ES)
                ↓
          Controller tested in Real Env (transfer)
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|---|---|---|
| Environment | CarRacing-v0 (Box2D, top-down car) | PointMassNav (numpy 2D arena) |
| VAE latent dim | 32 (same) | 32 |
| MDN mixtures | 5 (same) | 5 |
| RNN | LSTM (same) | LSTM, hidden=128 |
| Controller | Linear (z,h → a), CMA-ES | Linear, CMA-ES (or random search) |
| Observation | 64×64 color (3ch) | 64×64 grayscale (1ch) |
| Training scale | GPU, hours | CPU, minutes |
| Dream rollout reward | From real env proxy | Proxy reward from latent distance |

The algorithm and architecture are faithful; only the environment and scale are reduced for CPU trainability.
