# Improved Training of Wasserstein GANs (WGAN-GP)

**Paper:** [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
**Authors:** Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville
**Year:** 2017 (NeurIPS 2017)

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/improved-training-wasserstein-gans-gp)

## Summary

Improved Training of Wasserstein GANs (WGAN-GP) replaces the weight-clipping mechanism of the original WGAN with a **gradient penalty** to enforce the Lipschitz constraint on the critic. The original WGAN clamped critic weights to a bounded box \([-c, c]\) after each update, which led to pathological behavior: capacity underuse (the critic learns overly simple functions), exploding/vanishing gradients when the clipping threshold is poorly chosen, and slow convergence. WGAN-GP instead penalizes the critic when the gradient norm of its output with respect to its input deviates from 1, evaluated at random interpolations between real and generated samples. This softly enforces the 1-Lipschitz condition without the crude side effects of weight clipping.

The key practical change: instead of clamping weights, sample a random point \(\hat{x}\) on the line segment between a real sample \(x\) and a fake sample \(G(z)\), compute the gradient \(\nabla_{\hat{x}} C(\hat{x})\), and add a penalty \(\lambda \mathbb{E}[(\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1)^2]\) to the critic loss. This enables stable training across a wide variety of architectures — including 101-layer ResNets and language models — with almost no hyperparameter tuning.

## Core Idea

The Wasserstein distance requires the critic \(C\) to be 1-Lipschitz. Weight clipping enforces this crudely by restricting the parameter space. WGAN-GP enforces it **softly** via a regularization term:

$$\mathcal{L}_{\text{critic}} = \underbrace{\mathbb{E}_{x \sim \mathbb{P}_g}[C(x)] - \mathbb{E}_{x \sim \mathbb{P}_r}[C(x)]}_{\text{Wasserstein estimate}} + \underbrace{\lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}}[(\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1)^2]}_{\text{gradient penalty}}$$

where \(\hat{x}\) is sampled by linearly interpolating between real and fake points:

$$\hat{x} = \epsilon x + (1 - \epsilon) G(z), \quad \epsilon \sim U[0, 1]$$

**Algorithm (WGAN-GP):**
1. For each critic step:
   a. Sample real batch \(x \sim \mathbb{P}_r\) and noise \(z \sim p(z)\)
   b. Generate fake batch \(G(z)\)
   c. Sample \(\epsilon \sim U[0,1]\), compute interpolations \(\hat{x} = \epsilon x + (1-\epsilon) G(z)\)
   d. Compute critic output on real, fake, and interpolated points
   e. Compute gradient penalty: \(\lambda (\|\nabla_{\hat{x}} C(\hat{x})\|_2 - 1)^2\)
   f. Critic loss = \(C(\text{fake}) - C(\text{real}) + \text{gradient penalty}\)
   g. Backprop and step optimizer (Adam, lr=1e-4)
2. Repeat for \(n_{\text{critic}}\) steps (typically 5)
3. Update generator: minimize \(-C(G(z))\)

## Key Method Details

- **Gradient penalty instead of weight clipping:** The penalty \(\lambda (\|\nabla C\|_2 - 1)^2\) softly pushes the critic toward 1-Lipschitz behavior, where \(\lambda = 10\) is the recommended weight
- **Interpolation points:** The penalty is evaluated at random points on the line between real and fake samples — the optimal transport path — rather than uniformly over the data space
- **Adam optimizer:** WGAN-GP uses Adam with lr=1e-4, β1=0.5, β2=0.9 (unlike original WGAN which used RMSProp with lr=5e-5)
- **n_critic = 5:** Critic updated 5 times per generator update (same as WGAN)
- **No weight clipping:** All weights are free to move; the Lipschitz constraint is enforced only through the gradient penalty
- **Batch normalization removed from critic:** BN interferes with the gradient penalty because the per-sample gradient depends on the batch. Use LayerNorm or no normalization instead

## What Problem Does It Solve

Imagine you're teaching a robot to judge whether a painting looks real or fake. The robot needs to be a *fair* judge — it shouldn't give extreme scores like "infinity out of ten" or always say exactly the same thing. In the original WGAN, we kept the robot "fair" by putting it in a tiny box: we forced all its internal settings (weights) to be very small numbers between -0.01 and 0.01. This is like telling a food critic they can only use words from a tiny vocabulary — sure, they'll stay "fair," but they can barely express anything useful, and sometimes they get stuck giving the same boring review to everything.

WGAN-GP fixes this by letting the robot use its full vocabulary (no weight limits!) but instead teaching it a *rule*: "when you look at a painting, your reaction should change smoothly and at a reasonable rate — not too dramatic, not too flat." We check this by showing the robot a painting that's a blend between a real one and a fake one, and measuring how fast its opinion changes as we slowly morph from real to fake. If the opinion changes too fast or too slow, we give the robot a small penalty. This is like training a food critic to have *proportional* reactions — a slightly better dish should get a slightly better score, not a wildly different one. The result: the robot gives much more useful feedback to the painter, training is far more stable, and we don't need to fiddle with the "box size" (the clipping threshold) anymore.

## Influence

WGAN-GP became the **default training method for GANs** for several years, widely adopted in both research and industry. It was published at NeurIPS 2017 and has been cited over 8,000 times. The gradient penalty approach directly inspired subsequent Lipschitz enforcement methods, including **spectral normalization** (Miyato et al., 2018), which became the standard for SNGAN and many BigGAN variants. WGAN-GP loss is used as a key component in BigGAN, Progressive GAN, and numerous conditional GAN architectures. The paper's key insight — that enforcing smoothness via input-space gradients is better than constraining parameter space — influenced the broader study of regularization in generative models and the analysis of Lipschitz continuity in deep learning.

## arXiv Link

https://arxiv.org/abs/1704.00028
