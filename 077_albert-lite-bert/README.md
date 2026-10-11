# 077 — ALBERT: A Lite BERT for Self-supervised Learning of Language Representations

**Paper:** Lan, Z., Chen, M., Goodman, S., Gimpel, K., Sharma, P., & Soricut, R. (2020). *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. ICLR 2020. arXiv:1909.11942.

**arXiv:** https://arxiv.org/abs/1909.11942

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/albert-a-lite-bert-self-supervised)

---

## Summary

ALBERT (A Lite BERT) addresses the problem that increasing the size of pretrained language models quickly becomes infeasible due to GPU/TPU memory limits and prohibitively long training times. The paper introduces two parameter-reduction techniques — **factorized embedding parameterization** and **cross-layer parameter sharing** — that dramatically reduce the parameter count of BERT-style models without sacrificing (and in some cases improving) downstream performance. A third contribution is the replacement of BERT's Next Sentence Prediction (NSP) task with a **Sentence Order Prediction (SOP)** loss that models inter-sentence coherence, which consistently helps on multi-sentence downstream tasks. The best ALBERT configuration achieves state-of-the-art results on GLUE, RACE, and SQuAD while having significantly fewer parameters than BERT-large.

### Core Method Details

1. **Factorized Embedding Parameterization:** In standard BERT, the vocabulary embedding matrix has size V × H (vocabulary size × hidden dimension), which is enormous — e.g., 30,000 × 768 ≈ 23M parameters for just the embedding. ALBERT decomposes this into two smaller matrices: V × E (vocabulary → embedding dimension) and E × H (embedding → hidden dimension), where E is much smaller than H (e.g., E=128 vs H=768). This reduces the embedding parameters from O(V × H) to O(V × E + E × H), a 4–10× reduction with negligible impact on performance.

2. **Cross-Layer Parameter Sharing:** In standard BERT, each Transformer layer has its own set of parameters (attention weights, feed-forward weights). ALBERT shares all parameters across all layers — every layer computes the same transformation with the same weights. This reduces the parameter count linearly with the number of layers (a 12-layer model has ~12× fewer parameters in the encoder body). The paper also explores sharing only attention parameters or only FFN parameters, but full sharing works best. While the representations across layers become more similar (less diversity), the model still performs well and trains much faster.

3. **Sentence Order Prediction (SOP):** BERT's NSP task is a binary classification: does sentence B follow sentence A? ALBERT argues that NSP is too easy and conflates topic prediction with coherence prediction. SOP instead presents two consecutive sentences in either correct or swapped order, and the model must predict whether the order is correct. This forces the model to learn discourse-level coherence rather than topic matching. SOP consistently improves downstream performance on tasks involving sentence pairs.

4. **Scaling Benefits:** With the memory freed by parameter reduction, ALBERT can scale to much larger hidden dimensions (e.g., H=1024 or even H=2048) and more layers while keeping the total parameter count manageable. ALBERT-xxlarge (12 layers, H=4096, shared) has only ~235M parameters but significantly outperforms BERT-large (340M) on all benchmarks.

### Influence

ALBERT has been cited over 8,000 times and was published at ICLR 2020. It became a standard reference for parameter-efficient pretraining and influenced subsequent work on parameter sharing (e.g., Universal Transformers, Mixture-of-Experts routing). ALBERT models are available through Hugging Face Transformers and Google Research. The factorized embedding technique is now commonly used in efficient model architectures, and the SOP loss has been adopted in several subsequent pretraining studies.

---

## What problem does it solve

Imagine you're building a really big Lego castle. The bigger the castle, the more impressive it looks — but at some point, you run out of space on your table, and it takes so long to build that you lose interest before finishing. You wish there was a way to make the castle just as big and impressive but using fewer bricks and less time.

That's exactly the problem ALBERT solves for AI language models. BERT was a super-smart AI that learned to understand language by reading millions of sentences and guessing missing words (like a giant fill-in-the-blanks game). But to make BERT smarter, researchers kept making it bigger — more layers, more parameters — until it became so huge that most computers couldn't fit it in memory, and training it took days or weeks.

ALBERT said: "What if we build the same castle but reuse the same bricks across multiple floors?" Instead of giving each layer of the AI its own set of weights (like buying new bricks for every floor), ALBERT shares the same weights across all layers — like using one set of bricks and just rearranging them. It also made the word-lookup table (the biggest single piece) much smaller by splitting it into two smaller pieces instead of one giant one. And it swapped out a confusing "does this sentence come next?" game for a better "are these sentences in the right order?" game that actually teaches the AI how sentences flow.

The result? ALBERT has way fewer parameters than BERT, trains faster, uses less memory, and yet scores even higher on language tests. It proved that you don't always need a bigger model — sometimes you just need a smarter way to reuse what you have.
