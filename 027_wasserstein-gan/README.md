# Wasserstein GAN (WGAN)

**Paper:** [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
**Authors:** Martin Arjovsky, Soumith Chintala, Léon Bottou
**Year:** 2017

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/wasserstein-gan)

## Summary

Wasserstein GAN (WGAN) replaces the standard GAN discriminator with a *critic* that approximates the Wasserstein-1 (Earth Mover's) distance between the real and generated data distributions. Unlike the Jensen-Shannon divergence used in vanilla GANs—which saturates and provides vanishing gradients when the distributions are disjoint—the Wasserstein distance remains continuous and differentiable almost everywhere, even when the real and generated distributions have non-overlapping support. This yields more stable training, eliminates mode collapse in many settings, and produces a loss curve that meaningfully correlates with sample quality—something the original GAN loss fails to provide.

The key practical change is simple: remove the sigmoid from the discriminator output (making it a critic instead of a binary classifier), clamp the critic's weights to a bounded box \([-c, c]\) after each gradient update to enforce the Lipschitz constraint, and train the critic more steps per generator step (typically 5:1). The generator then minimizes the negative critic score.

## Core Idea

The Earth Mover's (Wasserstein) distance measures the minimum "cost" of transporting mass from one probability distribution to another, where cost = amount moved × distance moved. Formally:

$$W(\mathbb{P}_r, \mathbb{P}_g) = \inf_{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]$$

By the Kantorovich-Rubinstein duality, this equals:

$$W(\mathbb{P}_r, \mathbb{P}_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim \mathbb{P}_r}[f(x)] - \mathbb{E}_{x \sim \mathbb{P}_g}[f(x)]$$

where the supremum is over all 1-Lipschitz functions \(f\). The critic approximates this \(f\), and weight clipping enforces (a crude approximation of) the Lipschitz constraint.

**Algorithm (WGAN):**
1. Sample real batch \(x \sim \mathbb{P}_r\) and noise \(z \sim p(z)\)
2. Compute critic loss: \(\mathcal{L} = \frac{1}{m}\sum[f(x)] - \frac{1}{m}\sum[f(G(z))]\)
3. Update critic by ascending the gradient (Adam with lr ~5e-5)
4. Clamp weights \(\theta \in [-c, c]\) (typically \(c = 0.01\))
5. Repeat for \(n_{\text{critic}}\) steps (typically 5)
6. Sample noise, update generator by descending \(-\frac{1}{m}\sum f(G(z))\)

## Key Method Details

- **Critic (not discriminator):** No sigmoid, no binary classification—outputs a real-valued score
- **Weight clipping:** All critic parameters clamped to \([-0.01, 0.01]\) after each update
- **RMSProp or low-lr Adam:** Original paper used RMSProp with lr=5e-5; later works use Adam with low lr
- **n_critic = 5:** Critic updated 5 times per generator update
- **Meaningful loss:** The critic loss correlates with sample quality, enabling monitoring without external metrics

## What Problem Does It Solve

Imagine you have two piles of sand on a beach—one shaped like a star (real data) and one shaped like a blob (fake data). In a regular GAN, the "judge" tries to say "this is real" or "this is fake." But if the piles are far apart and don't overlap at all, the judge can easily tell them apart and gives up sending useful feedback to the sand-sculptor (the generator). The sculptor doesn't know *how* to improve—they just know they're wrong.

The Wasserstein GAN fixes this by measuring the *distance* between the piles instead of just saying "same" or "different." It's like asking "how much sand do I need to move, and how far, to turn the blob into the star?" Even when the piles are completely separate, this distance gives a smooth, continuous signal: "move 3 buckets of sand 5 feet to the left." The sculptor always gets useful directions, no matter how far off they are. This means training is more stable, the sculptor doesn't get stuck repeating the same mistake (mode collapse), and the distance score tells you *how good* the sculpture is getting—just like checking a thermometer to see if you're getting closer to the right temperature.

## Influence

WGAN was a landmark paper that directly led to the Improved Training of Wasserstein GANs (WGAN-GP) paper, which replaced weight clipping with a gradient penalty to enforce the Lipschitz constraint more effectively. The Wasserstein distance became the default training objective for many subsequent GAN architectures, including BigGAN and StyleGAN variants. The paper has been cited over 10,000 times and is considered one of the most important theoretical contributions to GAN training stability.

## arXiv Link

https://arxiv.org/abs/1701.07875
