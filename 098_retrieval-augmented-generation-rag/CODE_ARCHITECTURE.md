# Code Architecture — RAG (Retrieval-Augmented Generation)

## Overview

This notebook implements a compact end-to-end RAG pipeline that mirrors the core architecture from Lewis et al. (2020): a **dense retriever** that fetches relevant passages from a document index, and a **seq2seq generator** that produces grounded answers conditioned on both the query and retrieved passages. The implementation uses small pre-trained models to stay within Kaggle's GPU/time constraints while faithfully demonstrating the RAG-Sequence and RAG-Token formulations.

## Section-by-Section Breakdown

### 1. Environment Setup & Imports
- Installs `sentence-transformers`, `faiss-cpu`, and `transformers` via pip.
- Imports PyTorch, transformers, sentence-transformers, and FAISS.
- Sets device (CUDA if available, else CPU).

### 2. Toy Document Collection (Non-Parametric Memory)
- Creates a toy knowledge base of ~40 short Wikipedia-style passages spanning science, history, geography, and technology.
- Each passage is a self-contained "document" (analogous to the 100-token Wikipedia chunks used in the paper).
- This replaces the 21M-passage Wikipedia index with a manageable toy set that demonstrates the full pipeline.

### 3. Dense Retriever (DPR Analog)
- Uses `sentence-transformers/all-MiniLM-L6-v2` as the passage/question encoder (a compact BERT-based encoder, analogous to DPR's dual-BERT encoders).
- Encodes all documents into dense vectors (384-dim) and builds a FAISS index for fast Maximum Inner Product Search (MIPS).
- `retrieve(query, k=5)` returns the top-k passages with their similarity scores.
- **Deliberate simplification:** The original paper uses two separate encoders (one for questions, one for passages) trained with contrastive learning. We use a single pre-trained bi-encoder for simplicity — the retriever is not fine-tuned end-to-end with the generator (the toy scale makes joint fine-tuning impractical).

### 4. RAG Generator (BART Analog)
- Uses `google/flan-t5-base` (250M params) as the seq2seq generator — a smaller stand-in for BART-large (400M).
- The generator takes a formatted prompt: `"question: <query> context: <retrieved_passage>"` and generates an answer.
- This mirrors the paper's approach of concatenating the retrieved passage with the query as input to the seq2seq model.

### 5. RAG-Sequence Implementation
- Retrieves top-k passages for a query.
- For each retrieved passage, runs the generator independently to produce a candidate answer.
- Scores each candidate by combining retrieval probability p(z|x) (softmax of retrieval scores) and generation probability p(y|x,z) (generator log-probability).
- Returns the highest-scoring answer — this is the RAG-Sequence marginalization: argmax_y Σ_i p(z_i|x) · p(y|x, z_i).
- **Key function:** `rag_sequence_answer(query, k=5)` — returns the best answer, all candidates, and their scores.

### 6. RAG-Token Implementation
- Retrieves top-k passages for a query.
- Uses beam search where at each generation step, the next-token distribution is the weighted sum of per-document next-token distributions.
- This allows different documents to influence different tokens in the output.
- **Key function:** `rag_token_answer(query, k=5)` — returns the RAG-Token answer with per-step document contributions.
- **Deliberate simplification:** Full RAG-Token beam search requires custom beam search code that interleaves retriever weights at each step. We implement a greedy version that computes per-token weighted distributions, which captures the core idea.

### 7. Parametric-Only Baseline (No Retrieval)
- Runs the same generator with NO retrieved context (just the question).
- This is the "parametric-only" baseline from the paper — the model relies entirely on its weights.
- **Key function:** `parametric_answer(query)` — baseline for comparison.

### 8. Comparison & Evaluation
- Tests the pipeline on 5 knowledge-intensive questions.
- Prints answers from: (a) parametric-only, (b) RAG-Sequence, (c) RAG-Token.
- Shows which passages were retrieved for each query.
- Demonstrates that RAG produces more factual, grounded answers than the parametric baseline.

### 9. Retrieval Quality Analysis
- Visualizes retrieval scores for sample queries.
- Shows the top-5 retrieved passages with similarity scores.
- Demonstrates the retriever's ability to find relevant passages.

### 10. Knowledge Update Demonstration
- Shows that updating the document index (adding/removing passages) immediately changes answers without retraining.
- This demonstrates the key advantage of non-parametric memory.

## Key Functions/Classes

| Function | Purpose |
|---|---|
| `encode_documents(docs)` | Encodes passages into dense vectors, builds FAISS index |
| `retrieve(query, k=5)` | Top-k retrieval from the FAISS index |
| `format_prompt(query, passage)` | Formats query + passage as generator input |
| `rag_sequence_answer(query, k=5)` | RAG-Sequence: best of k per-document answers |
| `rag_token_answer(query, k=5)` | RAG-Token: per-token weighted marginalization |
| `parametric_answer(query)` | Baseline: generation without retrieval |
| `compute_logprob(generator, input_text, output_text)` | Computes log p(y|x,z) for scoring |

## Data Flow / Shapes

```
Query (str)
  → Retriever encoder → query_vec (1, 384)
  → FAISS search → top-k indices (k,) + scores (k,)
  → k retrieved passages (k, str)
  → For each passage:
      format_prompt(query, passage) → input_ids (1, seq_len)
      generator.generate() → answer (str), logprob (float)
  → RAG-Sequence: weight by softmax(retrieval_scores) → best answer
  → RAG-Token: per-token weighted logits → greedy decode
```

## Deliberate Simplifications vs Full Paper

1. **Retriever:** Single pre-trained bi-encoder (`all-MiniLM-L6-v2`) instead of DPR's dual-encoder with contrastive training. No end-to-end retriever fine-tuning.
2. **Generator:** `flan-t5-base` (250M) instead of `bart-large` (400M). Not fine-tuned on downstream tasks (the paper fine-tunes on NQ/TriviaQA/WebQ).
3. **Index:** ~40 toy passages instead of 21M Wikipedia passages. FAISS-cpu instead of FAISS-GPU.
4. **RAG-Token:** Greedy decoding instead of full beam search with per-step document marginalization (the concept is the same; beam search adds significant complexity for marginal quality gain at this scale).
5. **No end-to-end fine-tuning:** The paper jointly fine-tunes retriever + generator. We use off-the-shelf models and demonstrate the inference-time RAG pipeline, which is how RAG is most commonly deployed in practice today.
6. **Evaluation:** Qualitative comparison on 5 questions instead of quantitative benchmarks (NQ, TriviaQA, WebQuestions) which require full datasets and fine-tuning.
