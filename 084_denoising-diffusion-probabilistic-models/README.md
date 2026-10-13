# Denoising Diffusion Probabilistic Models (DDPM)

**Paper:** Ho, Jain, & Abbeel (2020). *Denoising Diffusion Probabilistic Models.* arXiv:2006.11239
**Authors:** Jonathan Ho, Ajay Jain, Pieter Abbeel (UC Berkeley)
**arXiv:** https://arxiv.org/abs/2006.11239

## Summary

Denoising Diffusion Probabilistic Models (DDPM) are a class of generative models that learn to produce data (e.g., images) by learning to reverse a gradual noising process. The forward (diffusion) process progressively adds Gaussian noise to an image over T timesteps until it becomes pure noise. A neural network — specifically a U-Net — is trained to predict the noise added at each timestep, enabling a reverse process that iteratively denoises from pure Gaussian noise back to a realistic image. The training objective simplifies to a mean squared error loss between predicted and actual noise. On CIFAR-10, DDPM achieved an FID of 3.17 (state-of-the-art at the time), rivaling and in some cases surpassing GANs in sample quality.

## Core Idea

The key insight is that while adding noise is trivial and closed-form (a Markov chain of Gaussian transitions), removing noise can be learned by a neural network parameterized as Gaussian transitions. The forward process q(x_t | x_{t-1}) adds noise according to a variance schedule {β_t}. Crucially, any intermediate noisy state x_t can be computed directly from x_0 in closed form: q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t)I), where ᾱ_t = ∏(1-β_i). This allows training on arbitrary timesteps without simulating the full chain.

The reverse process p_θ(x_{t-1} | x_t) is parameterized as a Gaussian with mean and variance predicted by a U-Net conditioned on the timestep t. The training objective reduces to a simple MSE loss: L = E[||ε - ε_θ(x_t, t)||²], where ε is the actual noise and ε_θ is the network's prediction.

## Key Method Details

- **Forward process:** T=1000 steps, linear variance schedule β_t from 1e-4 to 0.02
- **Closed-form noising:** x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε, where ε ~ N(0,I)
- **Reverse process parameterization:** The model predicts noise ε_θ(x_t, t), and the mean is computed analytically: μ_θ = (1/√(ᾱ_t)) (x_t - (β_t/√(1-ᾱ_t)) ε_θ)
- **U-Net architecture:** Encoder-decoder with skip connections, sinusoidal time embeddings, group normalization, and self-attention at intermediate resolutions
- **Training:** Random timestep sampling, simple MSE loss on predicted noise
- **Sampling:** Iterative denoising from x_T ~ N(0,I) through all T steps to x_0

## What Problem Does It Solve

Imagine you have a photograph and you slowly add TV static (noise) to it, bit by bit, until it's completely unrecognizable — just pure static. Now imagine teaching a computer to do the reverse: start with pure static and gradually remove the noise until a clear picture emerges. That's what DDPM does.

**Real-life example:** Think of it like restoring an old, damaged photograph. If someone hands you a photo that's been through years of wear — faded, scratched, covered in dust — you can probably guess what the original looked like because you've seen many photos before. DDPM trains a neural network the same way: by showing it thousands of examples of images at every stage of being "damaged" with noise, it learns to "restore" them. Once trained, you can hand it pure random static, and it will iteratively clean it up into a brand-new, realistic image that never existed before — like a face, a car, or a cat.

This solved a major problem in AI: how to generate high-quality images without using GANs (which are notoriously hard to train and prone to mode collapse). DDPMs offered stable training, diverse samples, and quality that matched or beat GANs.

## Influence

DDPM sparked the diffusion model revolution. Direct descendants include DDIM (faster sampling), Stable Diffusion (latent space diffusion for text-to-image), DALL-E 2, Imagen, and essentially all modern AI image generators. The paper has been cited thousands of times and is one of the most influential generative modeling papers of the decade.

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/084_denoising-diffusion-probabilistic-models)
