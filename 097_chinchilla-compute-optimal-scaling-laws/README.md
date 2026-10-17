# Training Compute-Optimal Large Language Models (Chinchilla)

**arXiv:** [2203.15556](https://arxiv.org/abs/2203.15556) | **Authors:** Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre (DeepMind) | **Year:** 2022

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/chinchilla-compute-optimal-scaling-laws)

## Summary

The Chinchilla paper by DeepMind fundamentally changed how we think about scaling large language models. Before this work, the dominant strategy was to make models bigger and bigger while keeping training data roughly constant — following the Kaplan et al. (2020) scaling laws, which suggested that model parameters should scale ~9x faster than training tokens. Hoffmann et al. trained over 400 models ranging from 70M to 16B parameters on 5 to 500 billion tokens and showed that this was wrong: existing large models were severely **undertrained**. The compute-optimal strategy scales parameters and data **equally** — for every doubling of model size, you should also double the training tokens. The authors proved this by training Chinchilla (70B parameters, 1.4 trillion tokens), which used the same compute budget as Gopher (280B) but with 4× more data. Chinchilla outperformed Gopher, GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) across every benchmark, achieving 67.5% on MMLU — a 7% improvement over Gopher. This paper reshaped the entire LLM industry: subsequent models (LLaMA, GPT-4, Falcon) all adopted the Chinchilla ratio of ~20 tokens per parameter.

## Core Idea

Given a fixed compute budget **C** (measured in FLOPs), the question is: how should you split that budget between model parameters **N** and training tokens **D** to minimize the loss **L(N, D)**? The paper approaches this with three complementary methods:

1. **Approach 1 — Fix model sizes, vary training tokens:** Train models at fixed sizes (70M to 16B) for different numbers of steps. For each compute level, find the model size that achieves the lowest loss. Result: N_opt ∝ C^0.50, D_opt ∝ C^0.50.

2. **Approach 2 — IsoFLOP profiles:** For each target compute budget, train models of various sizes such that they all consume the same total FLOPs. Fit the loss curves to find the optimal point. Result: N_opt ∝ C^0.49, D_opt ∝ C^0.51.

3. **Approach 3 — Parametric loss function:** Propose the functional form:

   > L̂(N, D) = E + A/N^α + B/D^β

   where E is the irreducible entropy of natural text, and the A and B terms capture the capacity and data bottlenecks respectively. Fitting over all training runs yields E=1.69, A=406.4, B=410.7, α=0.34, β=0.28. The optimal allocation derived from this fit: N_opt ∝ C^0.46, D_opt ∝ C^0.54.

All three approaches converge on the same conclusion: **parameters and data should scale in approximately equal proportions** — roughly N_opt ∝ C^0.5, D_opt ∝ C^0.5. This is in stark contrast to Kaplan et al. (2020), which predicted N_opt ∝ C^0.73, D_opt ∝ C^0.27 (i.e., 9× more parameters than data per unit compute).

The FLOP budget is approximated as C ≈ 6·N·D (6 FLOPs per parameter per token for forward + backward pass).

## Key Method Details

- **Models trained:** Over 400 transformer models, from 70M to 16B+ parameters, trained on 5B to 500B tokens from the MassiveText dataset.
- **Chinchilla model:** 70B parameters, trained on 1.4 trillion tokens, same compute budget as Gopher (~5.76×10^23 FLOPs). Used AdamW optimizer instead of Adam, otherwise same architecture and data as Gopher.
- **Scaling law formula:** L(N, D) = E + A/N^α + B/D^β with fitted parameters E=1.69, A=406.4, B=410.7, α=0.34, β=0.28.
- **Optimal compute allocation:** N_opt(C) ∝ C^a, D_opt(C) ∝ C^b where a ≈ b ≈ 0.5 (compared to Kaplan's a=0.73, b=0.27).
- **Tokens-to-parameters ratio:** ~20 tokens per parameter at compute-optimal training (Chinchilla: 1.4T tokens / 70B params ≈ 20).
- **Key finding:** Most existing large models (GPT-3, Gopher, Jurassic-1) were significantly undertrained — they used far fewer tokens than optimal for their parameter count.
- **Benchmarks:** Chinchilla achieved 67.5% average on MMLU (vs. Gopher's 60.0%), and outperformed on BIG-bench, reading comprehension, common sense, and closed-book QA.

## What problem does it solve

Imagine you have a fixed amount of money to bake the best possible cake. You can spend it on two things: a bigger oven (more capacity) or more ingredients (more data to practice with). Before this paper, everyone was buying bigger and bigger ovens but barely buying any extra ingredients. They'd have these massive, fancy ovens but only bake a few small cakes — the oven was wasted because it never got enough practice. 

This paper figured out the *perfect recipe*: for every dollar you spend on a bigger oven, you should spend an equal dollar on more ingredients. The researchers proved it by baking a cake with a medium oven but tons of ingredients (Chinchilla: 70B parameters, 1.4 trillion tokens) — and it tasted way better than cakes from bakeries with giant ovens but few ingredients (Gopher: 280B parameters, 300B tokens). After this paper, every AI bakery changed their recipe. That's why models like LLaMA and GPT-4 now use much more training data relative to their size — they all follow the Chinchilla rule.

## Influence

The Chinchilla paper is one of the most impactful LLM papers of the 2020s. It directly changed how every major AI lab trains models: LLaMA (Meta), Falcon (TII), GPT-4 (OpenAI), and Mistral all adopted the Chinchilla scaling ratio of ~20 tokens per parameter. The Kaplan scaling laws (which advised scaling parameters 9× faster than data) were effectively retired. The paper also had massive implications for data collection — if you need 20× more tokens than parameters, then a 70B model needs 1.4T tokens, which drove the race for high-quality training data (RefinedWeb, FineWeb, etc.). The parametric loss formula L(N,D) = E + A/N^α + B/D^β became the standard scaling-law fitting framework used in subsequent work (Llama-2, Llama-3, and Chinchilla-derived scaling laws for compute-optimal training).
