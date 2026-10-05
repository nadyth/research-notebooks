# Code Architecture — Neural Ordinary Differential Equations

## Notebook Overview

The notebook implements the three core contributions of the paper in a self-contained, from-scratch manner:

1. **ODE-Net classifier** — a continuous-depth network for 2D toy classification
2. **Adjoint method gradient computation** — manual implementation alongside `torchdiffeq`
3. **Continuous transformation visualization** — showing the flow of data through continuous time

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `torchdiffeq` (the official library from the paper authors)
- Imports PyTorch, NumPy, Matplotlib
- Sets random seeds for reproducibility

### 2. Toy 2D Dataset
- Generates two concentric circles (sklearn `make_circles`) or two moons
- This is the standard toy dataset for visualizing continuous transformations
- Shape: `(N, 2)` input features, `(N,)` binary labels
- **Deliberate simplification:** The paper uses MNIST; we use 2D data so the transformation can be directly visualized as a vector field / flow. MNIST ODE-Nets take 10+ minutes on GPU and are harder to visualize.

### 3. ODE Function (Vector Field)
- A small MLP `f_θ(h(t), t)` that takes the current state h(t) and time t, outputs dh/dt
- Architecture: `Linear(2→64) → Tanh → Linear(64→64) → Tanh → Linear(64→2)`
- Time `t` is concatenated to the state as an additional input dimension
- This is the "infinitely deep" network — each "layer" is a point in continuous time

### 4. ODE-Net Classifier
- **Forward pass:** `h(T) = odeint(f_θ, h(0), t_span)` where `t_span = [0, 1]`
- Uses the `dopri5` (Dormand-Prince) adaptive solver — the default in the paper
- After ODE integration, a linear classifier head maps h(T) → class logits
- Training loop: standard cross-entropy loss, Adam optimizer

### 5. Training
- **Key hyperparameters:**
  - Integration time: t ∈ [0, 1.0]
  - ODE solver: `dopri5` (adaptive, Runge-Kutta 4(5))
  - Tolerance: rtol=1e-3, atol=1e-3 (relaxed for speed)
  - Epochs: 100 (toy dataset converges fast)
  - Learning rate: 1e-3
- Loss: cross-entropy on 2-class problem
- **Deliberate simplification:** Paper uses full MNIST with downsampling + 6 residual blocks; we use 2D toy data for visualization and speed.

### 6. Visualizing the Continuous Transformation
- **Vector field plot:** For a grid of points in 2D space, evaluate f_θ(h, t=0.5) and draw arrows showing dh/dt
- **Flow / trajectory plot:** Pick sample points, integrate the ODE from t=0 to t=1 at many intermediate times, plot the trajectory of each point through the "continuous network"
- Shows how the initial data distribution (e.g., two interleaving circles) is continuously warped into a linearly separable configuration
- This is the key visualization from the paper (Figure 1 / Figure 5)

### 7. Adjoint Method (Manual Implementation)
- Implements the adjoint sensitivity method from scratch to show how gradients flow backward through the ODE
- **Forward:** h(T) = ODE solve forward
- **Backward:** Solve augmented ODE backward:
  - Adjoint: `da/dt = -a^T · ∂f/∂h`
  - Parameter gradient: `dL/dθ = -∫ a(t)^T · ∂f/∂θ dt`
- Uses `torch.autograd.Function` to create a custom differentiable ODE block
- Compares gradients from the manual adjoint vs `torchdiffeq`'s built-in adjoint mode

### 8. Memory Comparison: ODE-Net vs ResNet
- Constructs a matching ResNet (6 residual blocks) with the same width
- Measures peak memory during forward+backward for both
- Shows ODE-Net uses O(1) memory (constant regardless of solver steps) vs ResNet's O(L) where L = number of layers
- Uses `torch.cuda.memory_allocated()` or CPU memory tracking

### 9. Adaptive Solver Step Count
- Counts the number of function evaluations (NFE) the adaptive solver makes
- Shows that the solver adaptively uses more steps for harder inputs and fewer for easier ones
- This is a key property from the paper: "adapt their evaluation strategy to each input"

### 10. Continuous Normalizing Flow (Bonus)
- Implements the instantaneous change of variables: `d/dt log p(z(t)) = -Tr(∂f/∂z)`
- Uses Hutchinson's trace estimator to approximate the trace efficiently
- Trains a small CNF to transform a Gaussian into the two-moons distribution
- Visualizes the density evolution over continuous time

## Key Functions / Classes

| Name | Type | Purpose |
|---|---|---|
| `ODEFunc` | `nn.Module` | The vector field f_θ(h(t), t) — the "continuous network" |
| `ODEBlock` | `nn.Module` | Wraps `torchdiffeq.odeint` into a callable PyTorch layer; supports adjoint mode |
| `ODENet` | `nn.Module` | Full classifier: input → ODEBlock → linear head → logits |
| `AdjointODE` | `autograd.Function` | Manual adjoint method implementation for gradient computation |
| `compute_nfe` | function | Count number of forward evaluations of f_θ during odeint |
| `visualize_vector_field` | function | Plot the learned vector field f_θ on a 2D grid |
| `visualize_flow` | function | Plot sample trajectories through continuous time |

## Data Flow / Shapes

```
Input: (B, 2)           # 2D toy data points
  → ODEFunc(h, t): (B, 2+1) → (B, 64) → (B, 64) → (B, 2)   # vector field, t concatenated
  → odeint: (B, 2)      # integrated from t=0 to t=1
  → Linear(2→2): (B, 2)  # classifier head → logits
  → cross_entropy: scalar
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook | Rationale |
|---|---|---|---|
| Dataset | MNIST (28×28 images) | 2D toy (circles/moons) | Visualization + speed |
| Architecture | Downsample + 6 residual blocks → ODESolve | Single ODE block, 64-width MLP | Simplicity, <30 min on GPU |
| Applications | ODE-Net + CNF + Latent ODE | ODE-Net + vector field viz + CNF (bonus) | Focus on core concepts |
| Solver | dopri5, rtol=1e-5 | dopri5, rtol=1e-3 | Relaxed tolerance for speed |
| Adjoint | Full adjoint via augmented ODE | Manual adjoint + comparison with torchdiffeq | Pedagogical clarity |
| CNF trace estimator | Hutchinson's with exact trace comparison | Hutchinson's only | Simplicity |
