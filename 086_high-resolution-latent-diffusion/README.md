# High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)

**Paper:** Rombach, Blattmann, Lorenz, Esser, & Ommer (2022) — [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | CVPR 2022

Latent Diffusion Models (LDMs) move the diffusion process out of high-dimensional pixel space and into a compressed latent space learned by a pretrained autoencoder. A standard DDPM-style U-Net denoiser then operates on this much smaller latent representation, dramatically cutting both training and inference cost while preserving visual fidelity. The paper also introduces cross-attention layers into the U-Net, enabling conditioning on text, class labels, and spatial layouts — the architectural foundation that became Stable Diffusion. By choosing a modest downsampling factor (f=4 or f=8), the autoencoder retains enough spatial detail for high-fidelity reconstruction while reducing the dimensionality the diffusion model must learn by a factor of 8–64×.

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/high-resolution-latent-diffusion)

## What problem does it solve?

Imagine you want to paint a giant mural on a wall, but instead of painting directly on the wall (which is huge and takes forever), you first build a small scaled-down model of the wall, paint your design on that tiny model (which is much faster and easier), and then project it back up to full size with a magic projector that fills in all the fine details perfectly. That's what Latent Diffusion does: regular diffusion models generate images by adding and removing noise directly on full-size pixels — like painting every single pixel of a giant canvas one at a time. This is incredibly slow and expensive, often requiring hundreds of GPU-days just to train. LDMs first compress the image into a tiny "latent" code using an autoencoder (the small model), run the diffusion process there (painting on the small model), then decode back to full resolution (the magic projector). The result: same quality images, but trained and generated dramatically faster and cheaper.

## Core Idea

1. **Perceptual compression stage (Autoencoder):** A VQ-regularized or KL-regularized autoencoder compresses images from pixel space (H×W×3) to a latent space (H/f × W/f × C_z) with downsampling factor f (typically 4 or 8). The autoencoder is trained once and frozen — it provides a perceptually equivalent but much lower-dimensional representation.

2. **Diffusion in latent space (LDM):** A DDPM-style U-Net denoiser operates on the latent codes instead of pixels. Since the latent is f²× smaller spatially, each forward/backward pass is f² cheaper. Training converges faster because the model spends capacity on semantic structure rather than imperceptible high-frequency details.

3. **Noise schedule in latent space:** The diffusion noise schedule must be adapted to the latent space's different scale. The paper uses a shifted noise schedule based on the latent's signal-to-noise ratio, ensuring the diffusion process covers the right range of noise levels.

4. **Cross-attention conditioning:** To enable conditional generation (text-to-image, class-conditional, layout-to-image), the U-Net's ResBlocks are augmented with cross-attention layers. A domain-specific encoder (e.g., a text transformer) maps the conditioning input to a context sequence, and cross-attention allows the denoiser to attend to this context at each resolution. This is the key architectural innovation that makes flexible, multi-modal conditioning possible.

5. **Two-stage training pipeline:** First, train the autoencoder independently (perceptual compression). Second, train the diffusion U-Net on frozen latent codes (latent diffusion). The decoupling means the expensive autoencoder training happens once, and different diffusion models (for different tasks/conditions) can reuse it.

## Influence

This paper is the direct foundation of Stable Diffusion, one of the most impactful open-source AI releases in history. The latent diffusion approach democratized high-quality image generation — making it feasible to train and run text-to-image models on consumer GPUs. With thousands of citations, it spawned an entire ecosystem: Stable Diffusion 1.x/2.x/3, ControlNet, DreamBooth, LoRA fine-tuning, and countless downstream applications in art, design, and content creation. The CompVis/latent-diffusion GitHub repository became one of the most starred ML projects ever. The paper was published at CVPR 2022 and its authors (Rombach, Blattmann) later joined Stability AI to build Stable Diffusion.
