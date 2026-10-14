# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

**Paper:** Fedus, Zoph, Shazeer (2021) — JMLR
**arXiv:** [2101.03961](https://arxiv.org/abs/2101.03961)
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/088_switch-transformer)

## Summary

The Switch Transformer replaces the standard feed-forward network (FFN) in a Transformer with a **Mixture-of-Experts (MoE) layer** that routes each token to exactly **one** expert (top-1 routing), rather than the top-k experts used in prior MoE work. This simple change reduces routing computation, halves expert batch sizes, lowers communication costs, and delivers up to **7× faster pre-training** than the dense T5-Base model at the same FLOP budget. The authors scale to **1-trillion-parameter** models and demonstrate stable training with bfloat16 precision using techniques like router z-loss, load-balancing auxiliary loss, and residual initialization. The sparse model's gains can also be distilled back into a dense model, preserving ~30% of the quality improvement.

## Core Idea

Instead of every token passing through the same FFN, the Switch Transformer maintains **N expert FFNs** in parallel. A lightweight **router** (a single linear layer + softmax) computes a probability distribution over experts for each token and picks the single highest-probability expert. Only that expert processes the token, so the total computation per token stays constant (one FFN's worth) while the parameter count grows linearly with the number of experts. An **auxiliary load-balancing loss** (the dot product of the fraction of tokens routed to each expert and the average router probability for each expert) encourages even usage across experts, preventing collapse where most tokens go to a single expert.

## Key Method Details

- **Switch routing (top-1):** Each token is routed to exactly one expert — simpler and faster than top-2 MoE routing. The router produces logits `W_r · x`, applies softmax, and selects `argmax`.
- **Expert capacity factor:** Each expert's capacity = `(tokens_per_batch / num_experts) × capacity_factor`. Tokens exceeding capacity are dropped (passed through via residual connection). Capacity factor of 1.0–1.25 works best.
- **Load-balancing loss:** `L_aux = α · N · Σ(f_i · P_i)` where `f_i` = fraction of tokens routed to expert i, `P_i` = average router probability for expert i, `N` = number of experts, `α` = weight (typically 0.01).
- **Training stability:** Router z-loss (penalizing large logits before softmax), dropout in routing, and residual initialization of expert weights.
- **Distillation:** Sparse Switch models can be distilled into dense models, preserving ~30% of quality gains with up to 82% compression.

## What Problem Does It Solve

Imagine you have a big team of workers, and every time a task comes in, you give it to **everyone** on the team to work on together. That's how regular AI models work — every piece of input goes through every part of the brain. If you want the brain to be smarter, you make every part bigger, which means every task takes much more time and energy.

The Switch Transformer does something smarter: it has a **supervisor** (the router) that looks at each task and says, "You — go to worker #3. You — go to worker #7." Each task only goes to **one** worker, not all of them. The workers are called "experts." Now you can have a huge team (trillions of experts) without any single task taking longer, because each task still only gets handled by one person. The supervisor learns over time which worker is best for which kind of task. It's like a hospital where the receptionist routes each patient to exactly the right specialist instead of having every doctor see every patient.

## Influence

- **5,044 citations** (Semantic Scholar), **591 influential citations** — one of the most impactful MoE papers.
- Published in JMLR (Journal of Machine Learning Research), 2022.
- Directly influenced Mixtral (Mistral AI's open-weight MoE model), GShard, GLaM, and the modern wave of sparse expert models in production LLMs.
- The top-1 routing insight became the standard for most subsequent MoE architectures.

## Simplifications in This Notebook

- Toy-scale token classification task (not full language modeling on C4).
- Small vocabulary (~1000 tokens) and 4–8 experts (vs. 128 in the paper).
- Compares Switch Transformer (top-1 routing) vs. dense Transformer FFN on training speed and parameter efficiency.
- Single GPU/CPU implementation — no distributed all-to-all communication.
