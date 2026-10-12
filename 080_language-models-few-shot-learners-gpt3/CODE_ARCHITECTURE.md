# Code Architecture — GPT-3: Language Models are Few-Shot Learners

## Overview

This notebook demonstrates the **in-context learning** paradigm introduced by GPT-3, using a small pretrained language model (GPT-2) on a toy sentiment classification task. We replicate the key experimental finding: accuracy improves as the number of in-context examples (K) increases.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install `transformers` and `torch` (pip-installable).
- Import standard libraries: `torch`, `transformers`, `numpy`, `matplotlib`.

### 2. Load Pretrained Model
- Load GPT-2 (small, 124M parameters) via `GPT2LMHeadModel.from_pretrained('gpt2')`.
- This is our "language model" — a stand-in for the much larger GPT-3.
- The model is frozen (`model.eval()`); no gradient updates occur.

### 3. Toy Dataset Construction
- Build a small sentiment classification dataset with positive and negative sentences.
- Split into "demonstration pool" (for in-context examples) and "evaluation set" (for testing).
- Each sample: `(text, label)` where label ∈ {positive, negative}.

### 4. Prompt Construction
- **Zero-shot:** `Task: Classify the sentiment as positive or negative.\nSentence: {text}\nSentiment:`
- **Few-shot:** Prepend K labeled examples:
  ```
  Task: Classify the sentiment as positive or negative.
  Sentence: {text_1}
  Sentiment: {label_1}
  Sentence: {text_2}
  Sentiment: {label_2}
  ...
  Sentence: {query_text}
  Sentiment:
  ```
- The model is expected to generate "positive" or "negative" as the next token.

### 5. Inference Function
- For each query, construct the prompt, tokenize, and run a forward pass.
- Extract the logits for the next token and compare probabilities of "positive" vs "negative" tokens.
- Predict the label with higher probability.
- This simulates how GPT-3 performs few-shot inference — purely via next-token prediction.

### 6. Few-Shot Evaluation Sweep
- Evaluate at K = 0, 1, 2, 4, 8, 16 in-context examples.
- For each K, randomly sample K demonstration examples and evaluate on the full test set.
- Report accuracy for each K.

### 7. Visualization
- Plot accuracy vs. number of in-context examples (K).
- This replicates the key figure from the paper showing the scaling relationship.

## Key Functions/Classes

| Function | Purpose |
|---|---|
| `construct_prompt(examples, query)` | Builds the text prompt with K examples + query |
| `predict_sentiment(model, tokenizer, prompt)` | Runs forward pass, compares token probabilities |
| `evaluate_few_shot(model, tokenizer, demos, test_set, K)` | Evaluates accuracy at a given K |
| `run_scaling_sweep(model, tokenizer, demos, test_set, K_values)` | Runs the full K sweep and returns accuracies |

## Data Flow / Shapes

```
Input text (str) → Prompt construction (str) → Tokenization (tensor[1, seq_len])
  → GPT-2 forward (logits[1, seq_len, vocab_size]) → Extract last token logits
  → Compare P("positive") vs P("negative") → Predicted label (str)
```

## Deliberate Simplifications vs Full Paper

1. **Model size:** We use GPT-2 (124M) instead of GPT-3 (175B). The in-context learning effect is weaker at small scale but still observable.
2. **Single task:** We evaluate only sentiment classification. The paper evaluates 50+ tasks.
3. **Token-level comparison:** We compare probabilities of "positive"/"negative" tokens rather than free generation. This is a cleaner evaluation for small models.
4. **Small dataset:** We use ~50 hand-crafted sentences rather than standard benchmarks (e.g., SST-2).
5. **No few-shot fine-tuning comparison:** The paper compares against fine-tuned models; we focus solely on the in-context scaling effect.
6. **Single model size:** The paper sweeps across model sizes (125M to 175B). We use a single model and sweep over K only.
