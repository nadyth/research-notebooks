# Denoising Diffusion Implicit Models (DDIM)

**Paper:** Song, Meng, & Ermon (2021) — [arXiv:2010.02502](https://arxiv.org/abs/2010.02502) | ICLR 2021

Denoising Diffusion Implicit Models (DDIM) accelerate sampling from diffusion models by constructing a family of non-Markovian forward processes that share the same training objective as DDPMs but admit much faster reverse generative processes. The key insight is that while DDPMs require iterating through all T diffusion steps (often 1000), DDIM defines a deterministic, non-Markovian reverse process that can use an arbitrary subsequence of timesteps — typically 10–50 steps instead of 1000 — producing samples of comparable quality 10× to 50× faster. Crucially, the same trained noise-prediction network is reused; no retraining is needed. DDIM also enables semantically meaningful latent-space interpolation and near-perfect reconstruction of training images, capabilities absent in stochastic DDPM sampling.

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/denoising-diffusion-implicit-models-ddim)

## What problem does it solve?

Imagine you're baking a cake from a recipe, but the recipe has 1000 tiny steps — add a grain of sugar, stir once, add a grain of flour, stir once, repeat. That's how DDPM (the original diffusion model) makes pictures: it removes a tiny bit of noise 1000 times to turn pure static into a clear image. It works great, but it's incredibly slow — like spending hours stirring one grain at a time. DDIM is like discovering that you can skip most of those steps and instead do just 20 big scoops and stirs, and the cake still comes out almost exactly the same. The magic is that you don't need a new recipe — you use the exact same ingredients and instructions, just skip steps smartly. This makes generating images 10 to 50 times faster, which is the difference between waiting hours and waiting minutes.

## Core Idea

1. **Non-Markovian forward process:** DDPMs define a Markov chain that progressively adds noise. DDIM generalizes this to a non-Markovian process where the forward posterior q(x_{t-1}|x_t, x_0) depends on x_0, not just x_t. This doesn't change the marginal q(x_t|x_0), so the training objective (noise prediction MSE) stays identical.

2. **Deterministic reverse process:** By setting σ_t = 0 (no stochastic noise injection during sampling), the reverse process becomes fully deterministic given the initial noise x_T. This is the key to DDIM's consistency property — the same x_T always produces the same image.

3. **Accelerated sampling via subsequence:** Instead of stepping through all T timesteps, DDIM selects a subsequence τ = [τ_1, ..., τ_S] of S steps (e.g., S=20 out of T=100). The update rule jumps directly between these timesteps:
   - x_{τ_{i-1}} = sqrt(α_{τ_{i-1}}) · predicted_x0 + sqrt(1 - α_{τ_{i-1}}) · direction_pointing_to_x_t
   - where predicted_x0 = (x_t - sqrt(1-α_t) · ε_θ(x_t,t)) / sqrt(α_t)

4. **Latent interpolation:** Because DDIM sampling is deterministic, the initial noise x_T acts as a meaningful latent code. Interpolating between two x_T vectors produces semantically meaningful image interpolations.

5. **ODE connection:** The deterministic DDIM sampler corresponds to the Euler discretization of a probability-flow ODE, linking diffusion models to neural ODEs.

## Influence

DDIM became the default fast sampler for diffusion models and is a foundational component of Stable Diffusion, DALL-E 2, and virtually all modern diffusion-based image generators. The ODE interpretation directly inspired subsequent work on DPM-Solvers, consistency models, and rectified flow. With over 3,000 citations, it is one of the most impactful papers in generative AI.
