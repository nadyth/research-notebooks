# LoRA: Low-Rank Adaptation of Large Language Models

**arXiv:** [2106.09685](https://arxiv.org/abs/2106.09685) | **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft Corporation) | **Year:** 2021

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/095_lora-low-rank-adaptation)

## Summary

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that freezes the pre-trained model weights and injects small trainable rank-decomposition matrices into each Transformer layer. Instead of updating all parameters during fine-tuning — which for GPT-3 175B means storing and deploying a separate 175B-parameter copy per task — LoRA approximates the weight update ΔW as a product of two low-rank matrices B (d×r) and A (r×k), where r is much smaller than the original dimensions. This reduces trainable parameters by up to 10,000× and GPU memory by 3×, while matching or exceeding full fine-tuning quality on RoBERTa, DeBERTa, GPT-2, and GPT-3. Critically, the low-rank matrices can be merged back into the original weights at inference time, introducing zero additional latency — unlike adapter layers which add sequential computation.

## Core Idea

The key insight is that the *change* in weights during adaptation (ΔW) has a low "intrinsic rank." Even though the pre-trained weight matrix W₀ ∈ ℝ^(d×k) has full rank, the adaptation delta doesn't need to. LoRA constrains the update as:

> W = W₀ + ΔW = W₀ + BA

where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), and r ≪ min(d, k). Only A and B are trained; W₀ is frozen. A is initialized with a random Gaussian, B is initialized to zero, so ΔW = BA = 0 at the start. The forward pass becomes h = W₀x + BAx, and the update is scaled by α/r where α is a constant. At deployment, W = W₀ + BA is computed once, so inference is identical to the original model.

## Key Method Details

- **Rank r**: Surprisingly small values (r = 1, 2, 4, 8) suffice even when d = 12,288 for GPT-3. The paper shows r = 4 on query and value projections captures most of the adaptation.
- **Scaling factor α**: The update is scaled by α/r. Setting α to the first r tried avoids re-tuning when varying r.
- **Which weights to adapt**: The paper applies LoRA only to attention weights (Wq, Wk, Wv, Wo) and freezes the MLP modules. Adapting Wq and Wv together with r = 4 is the recommended configuration.
- **Initialization**: A ~ N(0, σ²), B = 0. This ensures the model starts identical to the pre-trained model.
- **No inference latency**: Unlike adapters, LoRA weights can be merged into W₀, so the deployed model has the exact same architecture as the pre-trained model.
- **Task switching**: Swap BA for B'A' to switch tasks — only a few MB per task vs. hundreds of GB for full fine-tuning copies.

## What problem does it solve

Imagine you have a giant robot that's been trained to do a little bit of everything — it can talk, answer questions, and write stories. But now you want it to be really good at one specific job, like translating medical reports. The normal way is to *re-train the entire robot* for the medical job — but the robot is enormous (think a skyscraper-sized machine), and you'd need a whole new skyscraper for every single job you give it. That's super expensive and wasteful!

LoRA solves this by saying: "What if we don't retrain the whole robot? What if we just add a tiny backpack to it?" The backpack is so small it's like a wallet — it only changes a few things about how the robot thinks, but that's enough to make it great at the medical job. When you want the robot to do a *different* job instead, you just swap the wallet for a different one. The giant robot stays exactly the same. This means you can have one robot and a drawer full of wallets, instead of a hundred giant robots!

## Influence

LoRA has become one of the most widely adopted techniques in modern AI. It is the backbone of the HuggingFace PEFT (Parameter-Efficient Fine-Tuning) library, is used for fine-tuning Stable Diffusion models (Civitai hosts thousands of LoRA adapters), and is standard practice for customizing open-weight LLMs (Llama, Mistral, etc.) on consumer hardware. The paper has been cited thousands of times and spawned variants like QLoRA (quantized LoRA), LoRA+, and DoRA. It democratized LLM fine-tuning by making it possible on single GPUs with limited memory.
