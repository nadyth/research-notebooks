# 074 — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**Paper:** Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019.

**arXiv:** https://arxiv.org/abs/1810.04805

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/bert-pre-training-of-deep-bidirectional-transforme)

---

## Summary

BERT (Bidirectional Encoder Representations from Transformers) revolutionised natural language processing by introducing a pre-training method that produces deeply bidirectional language representations. Unlike previous approaches — GPT, which uses a left-to-right (unidirectional) Transformer decoder, and ELMo, which concatenates independently trained left-to-right and right-to-left LSTMs (shallow bidirectionality) — BERT uses a Transformer encoder where every token attends to both its left and right context simultaneously at every layer. The key innovation that enables this is the **Masked Language Model (MLM)** pre-training objective, inspired by the Cloze task: randomly mask 15% of input tokens and predict them based on the full bidirectional context. A second pre-training task, **Next Sentence Prediction (NSP)**, trains the model to understand relationships between sentence pairs. After pre-training on large unlabelled corpora, BERT can be fine-tuned on downstream tasks (classification, QA, NER) by adding a single task-specific output layer and updating all parameters — achieving state-of-the-art on 11 NLP benchmarks including GLUE (80.5%), MultiNLI (86.7%), SQuAD v1.1 (93.2 F1), and SQuAD v2.0 (83.1 F1).

### Core Method Details

1. **Transformer Encoder Architecture:** BERT uses a multi-layer bidirectional Transformer encoder (12 layers, 768 hidden, 12 attention heads for BERT-Base; 24 layers, 1024 hidden, 16 heads for BERT-Large). Every token can attend to all other tokens in both directions at every layer — true deep bidirectionality.

2. **Input Representation:** Input embeddings are the sum of token embeddings, segment embeddings (sentence A vs B), and position embeddings. A special `[CLS]` token is prepended for classification; `[SEP]` separates sentence pairs. WordPiece tokenisation (30k vocabulary) is used in the original paper; our implementation uses a simpler word-level vocabulary.

3. **Masked Language Model (MLM):** 15% of tokens are selected for masking. Of those, 80% are replaced with `[MASK]`, 10% with a random token, and 10% left unchanged. The model predicts the original token using cross-entropy loss. This forces the model to learn bidirectional context.

4. **Next Sentence Prediction (NSP):** Given sentence pairs (A, B), the model predicts whether B follows A. 50% of pairs are real consecutive sentences, 50% are random. The `[CLS]` representation is used for this binary classification.

5. **Fine-tuning:** After pre-training, the same model is fine-tuned on downstream tasks by adding a task-specific output head and training end-to-end on labelled data. This requires minimal architecture changes and is computationally inexpensive relative to pre-training.

### Influence

With over 130,000 citations, BERT is one of the most influential papers in NLP history. It set the paradigm of "pre-train on unlabelled text, fine-tune on labelled task data" that underpins virtually all modern NLP, including GPT-3, T5, and RoBERTa. BERT was open-sourced by Google AI Language in October 2018, with pre-trained models released alongside the paper, enabling immediate adoption by the research community. The paper was presented at NAACL 2019 and won the Best Long Paper Award.

---

## What problem does it solve

Imagine you're reading a fill-in-the-blank test: "The cat sat on the ___ because it was tired." To figure out that the blank is "mat" or "floor," you need to read both the words before AND after the blank. But older AI systems could only read left-to-right — like reading with a blindfold on your right eye. They'd see "The cat sat on the" but couldn't peek ahead at "because it was tired" to confirm their guess.

BERT fixes this by reading the whole sentence at once — both directions at the same time, like how you actually read. It randomly covers up some words and tries to guess them using ALL the surrounding words, not just the ones to the left. This makes BERT much better at truly understanding language, which means it can then be quickly taught to do tons of different tasks — like figuring out if a review is positive or negative, answering questions about a passage, or figuring out if two sentences mean the same thing — just by adding a small "answer button" on top and showing it a few examples. Before BERT, each of these tasks needed its own specially-designed system from scratch.
