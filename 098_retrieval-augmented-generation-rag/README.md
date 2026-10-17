# Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG)

**arXiv:** [2005.11401](https://arxiv.org/abs/2005.11401) | **Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela (Facebook AI Research, University College London) | **Year:** 2020

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/retrieval-augmented-generation-rag)

## Summary

The RAG paper introduced the concept of Retrieval-Augmented Generation — combining a pre-trained parametric language model (a seq2seq generator) with a non-parametric memory (a dense vector index of Wikipedia accessed via a pre-trained neural retriever). Instead of storing all factual knowledge inside the model's weights, RAG retrieves relevant passages from an external knowledge source at inference time and conditions the generator on both the input question and the retrieved passages. The paper proposes two formulations: **RAG-Sequence**, where the same set of retrieved documents conditions the entire generated sequence, and **RAG-Token**, where different documents can condition different tokens in the output. Both use a learned marginalization over retrieved documents — the generator produces outputs for each retrieved document, and the final distribution is a weighted sum over documents. RAG set state-of-the-art on three open-domain QA benchmarks (Natural Questions, TriviaQA, WebQuestions) and produced more specific, diverse, and factual text on generation tasks compared to parametric-only baselines. The term "RAG" has since become one of the most widely used concepts in modern AI — referring to the general pattern of augmenting LLMs with external retrieval.

## Core Idea

Traditional language models store knowledge in their parameters (weights). This has problems: knowledge is hard to update, models hallucinate, and they cannot provide provenance for their answers. RAG solves this by splitting knowledge into two parts:

1. **Parametric memory** — a pre-trained seq2seq model (the generator, originally BART) that handles language understanding and generation.
2. **Non-parametric memory** — a dense vector index of Wikipedia passages, accessed through a pre-trained neural retriever (DPR — Dense Passage Retriever).

At inference time, given a query **x**:
- The retriever selects the top-k most relevant passages **z** from the index
- The generator produces an output **y** conditioned on both **x** and the retrieved **z**
- The final output is marginalized over all retrieved documents: **p(y|x) = Σ p(z|x) · p(y|x,z)**

**RAG-Sequence** uses the same retrieved document for the entire output sequence: p(y|x,z_i) is computed per-document and summed. **RAG-Token** allows different documents to condition different tokens: p(y_j|x,z_i) is computed per-token, and the marginalization happens at each generation step.

The key insight is that the retriever and generator are jointly fine-tuned end-to-end — the retriever learns to fetch passages that help the generator, and the generator learns to use retrieved context effectively. The document index itself is kept fixed (not updated during training), which keeps the approach practical.

## Key Method Details

- **Retriever:** DPR (Dense Passage Retriever) — two BERT-based encoders, one for questions and one for passages, trained with contrastive learning to map relevant question-passage pairs close in vector space.
- **Generator:** BART-large (seq2seq transformer with 400M parameters) — takes the concatenation of the retrieved passage and the question as input, generates the answer.
- **Index:** Dense vector index of Wikipedia (21M passages from DPR's pre-built index), accessed via MIPS (Maximum Inner Product Search) using FAISS.
- **RAG-Sequence:** p(y|x) = Σ_i p(z_i|x) · p(y|x, z_i) — one document conditions the whole sequence.
- **RAG-Token:** p(y|x) = Π_j Σ_i p(z_i|x) · p(y_j|x, z_i) — different documents can condition different tokens.
- **Top-k retrieval:** Typically k=5 retrieved passages, with learned marginalization over all k.
- **End-to-end fine-tuning:** Both retriever and generator are fine-tuned jointly on downstream tasks (open-domain QA: Natural Questions, TriviaQA, WebQuestions; generation: MS MARCO, Jeopardy question generation).
- **Knowledge updating:** Because the index is non-parametric, knowledge can be updated by simply swapping or adding documents — no model retraining needed.
- **Results:** SOTA on Natural Questions (EM 44.5), TriviaQA (EM 56.8), WebQuestions (EM 41.5). Outperformed both parametric seq2seq baselines and task-specific retrieve-and-extract architectures.

## What problem does it solve

Imagine you're taking an open-book exam. You could try to memorize every fact in the textbook beforehand (that's how regular AI models work — they stuff everything into their "brain"/weights). But that's impractical: you can't memorize everything, facts change over time, and you can't show your teacher where you got each answer. 

Instead, imagine being allowed to bring a searchable encyclopedia to the exam. When you get a question, you look up the relevant pages, read them, and then write your answer based on both what you already know AND what you just looked up. That's exactly what RAG does. The AI model has a general understanding of language (like your brain), but when it needs a specific fact, it "looks up" relevant passages from a huge searchable database (the encyclopedia), reads them, and generates its answer using that fresh information. If a fact changes — like who the current president is — you just update the encyclopedia instead of retraining the entire AI. And you can always point to which page you used, so there's provenance. This paper started the whole "RAG" revolution that now powers things like AI chatbots that can search the web and cite their sources.

## Influence

The RAG paper coined the term that has become ubiquitous in AI. "RAG" is now a standard technique used across virtually every LLM application — from enterprise chatbots (Microsoft Copilot, Google Vertex AI Search) to coding assistants to web-search-augmented chat (Perplexity, ChatGPT's web browsing). The paper has been cited over 10,000 times and is considered foundational to the field of retrieval-augmented LLMs. The key insight — separating parametric knowledge (model weights) from non-parametric knowledge (external index) — has influenced architecture design across the industry. Subsequent work expanded on RAG: RETRO (DeepMind, scaling retrieval to trillions of tokens), Atlas (retrieval-augmented few-shot learning), and various RAG frameworks (LlamaIndex, LangChain, Haystack) all build on this paper's core formulation. The joint retriever-generator fine-tuning approach has been refined but the fundamental recipe remains: retrieve, then generate.
