# Layer Normalization

**Paper:** Ba, Kiros, Hinton (2016). *Layer Normalization.* arXiv:1607.06450
**Link:** https://arxiv.org/abs/1607.06450

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/layer-normalization)

## Summary

Layer Normalization (LayerNorm) is a normalization technique that computes the mean and variance used for normalization from **all the summed inputs to the neurons in a layer on a single training case**, rather than across a mini-batch of training cases as in Batch Normalization. This transposition of the normalization axis means LayerNorm is independent of batch size, works identically at training and test time, and can be applied straightforwardly to recurrent neural networks (RNNs) — computing normalization statistics separately at each time step. Like BatchNorm, each neuron gets its own adaptive bias and gain applied after normalization but before the non-linearity. The paper demonstrates that LayerNorm substantially reduces training time and stabilizes hidden-state dynamics in recurrent networks across six tasks: image-sentence ranking, question answering, language modelling, generative modelling, handwriting generation, and MNIST classification. LayerNorm became a foundational component of the Transformer architecture and nearly all modern large language models.

**Core idea:** Instead of normalizing across the batch dimension (BatchNorm), normalize across the feature dimension within each layer for each individual sample. This makes normalization independent of other samples in the batch.

**Key method details:**
- For a layer with H hidden units, compute μ = (1/H)Σ aᵢ and σ = √((1/H)Σ(aᵢ - μ)²) over all H summed inputs in the layer
- Normalize: āᵢ = (gᵢ/σ)(aᵢ - μ) + bᵢ, where g and b are learnable gain and bias
- Same computation at train and test time (no running averages needed)
- For RNNs: normalization statistics computed at each time step independently
- Invariant to re-scaling of the entire weight matrix and re-centering of all incoming weights
- Invariant to re-scaling of individual training cases

**Influence:** LayerNorm is a core component of the Transformer (Vaswani et al., 2017) and is used in virtually every modern transformer-based architecture including BERT, GPT, T5, and all contemporary large language models. It is one of the most cited normalization techniques in deep learning.

## What problem does it solve?

Imagine you're cooking a meal with many ingredients. If one ingredient is way too salty or way too bland, the whole dish tastes wrong — even if everything else is perfect. The same thing happens inside a neural network: some "neurons" (think of them as ingredients) might produce values that are way too big or way too small, and this throws off the rest of the network's calculations.

Before this paper, a popular fix called "Batch Normalization" solved this by comparing each ingredient to the same ingredient across many dishes (a batch of examples). But this only works well when you're cooking many dishes at once, and it's really hard to use when the "recipe" changes length — like translating sentences of different lengths.

Layer Normalization fixes this by checking all the ingredients **within a single dish** (one example) and making sure none of them is too extreme. This means:
- You can cook one dish at a time (batch size doesn't matter)
- The dish tastes the same whether you're practicing or serving to guests (same computation at train and test time)
- It works great even when recipes vary in length (perfect for language and sequence tasks)

This simple change made neural networks train faster and more stably, especially for text and language tasks, and became a standard ingredient in almost every AI model built today.

## Kaggle

[Open in Kaggle](https://www.kaggle.com/code/nadymsazad/layer-normalization)
