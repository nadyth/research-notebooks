# Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

**arXiv:** [1701.06538](https://arxiv.org/abs/1701.06538) (January 2017)

**Authors:** Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, Jeff Dean (Google Brain)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/sparsely-gated-mixture-of-experts-layer)

## Summary

This paper introduces the **Sparsely-Gated Mixture-of-Experts (MoE) layer**, a neural network component that dramatically increases model capacity without proportional increases in computation. The MoE layer consists of numerous feed-forward "expert" sub-networks and a trainable gating network that selects only a sparse subset of experts (top-k) to process each input. By activating just k out of n experts per example, the model gains capacity from all n experts' parameters while only paying the computational cost of k. The authors achieve greater than 1000× improvements in model capacity with minor computational overhead, reaching up to 137 billion parameters in models applied to language modeling and machine translation tasks.

## Core Idea

The key innovation is **Noisy Top-K Gating**: the gating network computes a score for each expert by projecting the input through a weight matrix, adding tunable Gaussian noise (controlled by a separate trainable weight matrix), then keeping only the top-k scores and setting the rest to −∞ before applying softmax. This produces a sparse gating vector where only k experts have non-zero weights. The MoE output is the weighted sum of the selected experts' outputs: y = Σ G(x)_i · E_i(x).

To prevent the gating network from collapsing onto a few favored experts (a self-reinforcing problem), the paper introduces two auxiliary losses:
- **Importance loss**: encourages all experts to receive equal total gate weight across a batch (coefficient of variation squared)
- **Load loss**: encourages balanced distribution of examples across experts, accounting for the noisy gating

The MoE layer is applied convolutionally between stacked LSTM layers — called once per time step, with potentially different experts selected at each position.

## Key Method Details

- **Expert architecture**: feed-forward networks with one hidden layer (ReLU activation), identical structure but separate parameters
- **Gating**: Noisy Top-K Gating with trainable noise via Softplus activation: H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)
- **Sparsity**: only top-k experts evaluated per example (k=4 in experiments, with up to 4096 experts)
- **Load balancing**: importance loss (CV² of batch-wise gate sums) + load loss (accounts for noisy gating probability)
- **Scaling**: mixing data parallelism and model parallelism — experts distributed across devices, standard layers replicated
- **Results**: 24% perplexity reduction on 1-Billion-Word benchmark; 4-billion parameter models outperform 151M parameter baselines at 6% of the compute

## What Problem Does It Solve

Imagine you run a big hospital and you have 1,000 different specialist doctors (experts). Every patient who walks in doesn't need to see all 1,000 doctors — that would be way too slow and expensive. Instead, you have a smart receptionist (the gating network) who quickly looks at each patient and says, "You only need to see 4 of our 1,000 doctors." The receptionist picks the *best* 4 doctors for that specific patient. This way, the hospital has the knowledge of 1,000 doctors available, but each patient only takes up the time of 4. That's exactly what this paper does for neural networks: instead of using all the parameters for every input, it keeps a huge pool of "expert" sub-networks but only activates a small handful for each piece of data. This lets the model be incredibly smart (have lots of parameters) without being incredibly slow (only compute with a few experts at a time).

## Influence

This paper is one of the foundational works behind modern large language models. The MoE concept directly inspired:
- **Switch Transformer** (Google, 2021) — simplified MoE with single-expert routing, scaling to trillion parameters
- **GShard** (Google, 2020) — MoE for multilingual translation at scale
- **GLaM** (Google, 2022) — 1.2T parameter MoE model
- **Mixtral 8x7B** (Mistral AI, 2023) — open-source MoE LLM
- **DeepSeek-MoE** and many modern LLM architectures that use sparse expert routing

The paper has been cited 2,500+ times and is considered the breakthrough that made conditional computation practical at scale.
