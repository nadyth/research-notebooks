# Neural Ordinary Differential Equations

**Paper:** Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. *Neural Ordinary Differential Equations*. NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/neural-ordinary-differential-equations)

## Summary

Neural ODEs replace the discrete stack of layers in a standard deep network with a *continuous* transformation parameterized by an ordinary differential equation (ODE). Instead of specifying N separate hidden layers, the model defines a vector field f_θ(h(t), t) that governs how the hidden state h evolves continuously from an initial value h(0) to a final value h(T). The output is computed by a numerical ODE solver (e.g., Dormand-Prince / `dopri5`) that adaptively decides how many function evaluations are needed. Training uses the **adjoint sensitivity method** (Pontryagin, 1962): rather than backpropagating through every internal step of the solver, gradients are computed by solving a second, augmented ODE *backward* in time. This gives O(1) memory cost regardless of how many solver steps were taken in the forward pass.

The paper demonstrates three applications: (1) **ODE-Net** — a continuous-depth residual network for MNIST classification that matches ResNet accuracy with fewer parameters and constant memory; (2) **continuous normalizing flows (CNFs)** — a generative model that computes exact log-likelihoods via the instantaneous change of variables formula, avoiding the need to partition or order data dimensions; and (3) **latent ODE time-series models** — a generative model for irregularly-sampled time series that encodes observations into a single latent initial state and decodes a continuous trajectory that can be evaluated at any time.

### Core Idea

The key insight is that a ResNet's forward pass — h_{l+1} = h_l + f(h_l, θ_l) — is exactly the Euler discretization of the ODE dh/dt = f(h(t), t, θ). Taking the limit of infinitely many infinitesimally small steps yields a continuous-depth network. The adjoint method makes this trainable: instead of storing all intermediate activations (which would be infeasible for an adaptive solver that may take hundreds of steps), it reconstructs gradients by integrating an augmented system backward from the loss to the initial state and parameters.

### Key Method Details

- **Forward pass:** Solve the IVP h(T) = h(0) + ∫₀ᵀ f(h(t), t, θ) dt using an adaptive ODE solver (e.g., `torchdiffeq.odeint` with the `dopri5` method).
- **Backward pass (adjoint):** Define the adjoint state a(t) = ∂L/∂h(t). It evolves backward via da/dt = −aᵀ ∂f/∂h. Parameter gradients follow dL/dθ = −∫ a(t)ᵀ ∂f/∂θ dt. All three quantities (a, dL/dθ, dL/dt₀) are obtained from a single backward ODE solve of an augmented system.
- **Instantaneous change of variables for CNFs:** For a continuous flow dz/dt = f(z(t), t), the log-density changes as d/dt log p(z(t)) = −Tr(∂f/∂z). This replaces the expensive log-det-Jacobian computation of discrete normalizing flows with a trace estimator (Hutchinson's).
- **Latent ODE for time series:** An RNN encoder maps irregularly-timed observations to a posterior over the initial latent state z(t₀); the ODE solver propagates z forward; a decoder reconstructs observations at any time. Extrapolation is trivial — just integrate further.

### What problem does it solve?

Imagine you're filling a glass of water from a tap. A regular neural network is like checking the water level only at fixed moments — after 1 second, after 2 seconds, after 3 seconds — and making a decision at each checkpoint. But what if the water flows at different speeds, or you need to know the level at 2.7 seconds? You'd have to guess between your checkpoints.

A Neural ODE is like having a smooth, continuous understanding of the water rising at every single instant — not just at fixed checkpoints. Instead of stacking dozens of separate processing layers (like Legos), it defines a smooth, continuous flow of information, like a river. A mathematical "ODE solver" figures out exactly how many checkpoints it needs based on how complicated the flow is — more checkpoints for tricky parts, fewer for simple parts.

The clever trick is the "adjoint method" for learning: instead of remembering every single checkpoint to learn from mistakes (which would use too much memory), it works *backward* from the final answer, reconstructing what went wrong step by step. This means the network can be as deep as it wants — hundreds of "layers" — and still use the same fixed amount of memory.

This solves real problems: modeling patient health data recorded at irregular intervals (not every hour, but whenever the nurse checks), predicting stock prices at arbitrary future times, and building generative models that can transform data distributions smoothly and reversibly.

### Influence

With over 11,000 citations on Google Scholar, Neural ODEs pioneered the field of **continuous-depth learning**. The paper won the **Best Paper Award at NeurIPS 2018**. It spawned numerous follow-ups: Neural SDEs (Kidger et al., 2020), FFJORD (Grathwohl et al., 2019) for scalable continuous normalizing flows, Neural CDEs (Kidger et al., 2020) for time series, and the `torchdiffeq` library became a standard PyTorch tool. The framework found applications in physics-informed neural networks, optimal control, and dynamical systems modeling, bridging deep learning with centuries of ODE theory.
