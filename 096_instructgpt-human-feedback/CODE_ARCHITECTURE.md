# Code Architecture: InstructGPT Notebook

## Overview

The notebook implements a simplified but complete RLHF (Reinforcement Learning from Human Feedback) pipeline on a small GPT-style language model. It demonstrates all three steps from the InstructGPT paper: (1) supervised fine-tuning (SFT) on instruction-response pairs, (2) reward model (RM) training on human preference rankings, and (3) PPO policy optimization against the reward model. The notebook compares outputs before and after RLHF to show the alignment effect.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import PyTorch, torch.nn, torch.nn.functional, and standard utilities.
- Set random seeds for reproducibility.
- Configure device (GPU/CPU).
- Install any pip dependencies (transformers tokenizers if needed).

### 2. Toy Instruction Dataset
- Create a small synthetic dataset of instruction-response pairs (e.g., "Write a haiku about rain" → "Gentle drops fall down / Pattering on rooftops soft / Earth drinks deeply now").
- Build a simple character or word-level tokenizer (kept minimal for self-containment).
- **Data shapes**: Input tokens `(batch, seq_len)`, target tokens `(batch, seq_len)`, attention mask `(batch, seq_len)`.
- For the RM dataset: prompt + K response candidates with rank labels `(batch, K, seq_len)`.

### 3. Small GPT-Style Language Model (Pretrained Base)
- **Key class**: `MiniGPT(nn.Module)` — a minimal GPT decoder with token embeddings, positional embeddings, N Transformer decoder blocks (causal multi-head self-attention + feedforward + residual + layer norm), and an LM head.
- Configurable `d_model=128`, `n_heads=4`, `n_layers=4`, `vocab_size`, `max_seq_len=128`.
- This simulates the "pretrained" GPT-3 — we initialize it and do a brief unsupervised pretraining pass on a small text corpus so it has non-random weights.
- **Forward shapes**: `(batch, seq_len)` → embeddings `(batch, seq_len, d_model)` → transformer blocks `(batch, seq_len, d_model)` → LM head `(batch, seq_len, vocab_size)`.

### 4. Step 1 — Supervised Fine-Tuning (SFT)
- Fine-tune MiniGPT on the instruction-response pairs using standard language modeling cross-entropy loss.
- Training: 20 epochs, cosine LR decay, dropout 0.1 (simplified from the paper's 16 epochs / dropout 0.2 for faster convergence on toy data).
- Model selection: keep the checkpoint with lowest validation loss.
- **Loss**: standard next-token CE: `L_SFT = −Σ log P(y_t | y_<t, x)`.
- **Output**: `sft_model` — the fine-tuned model used as initialization for both RM and PPO.

### 5. Step 2 — Reward Model (RM) Training
- **Key class**: `RewardModel(nn.Module)` — wraps the SFT model, replaces the LM head with a scalar output head (linear layer → single value).
- Input: prompt + response concatenated; output: scalar reward score.
- Training data: for each prompt, K=4 responses from the SFT model (with varying quality — some are good, some are deliberately degraded). A "human" ranking is simulated by scoring responses using a heuristic (e.g., response length, keyword matching, or a held-out quality function).
- **Loss** (pairwise ranking, from the paper):
  ```
  loss = −1/C(K,2) · Σ log(σ(r(x, y_w) − r(x, y_l)))
  ```
  All C(K,2) pairs from one prompt are computed in a single batch element.
- **Normalization**: After training, normalize RM so that SFT demonstrations achieve mean reward 0.
- **Output**: `reward_model` — used as reward function and value function for PPO.

### 6. Step 3 — PPO Policy Optimization
- **Key class**: `PPOTrainer` — orchestrates the PPO training loop.
- The RL policy is initialized from the SFT model. The environment is a bandit: sample a prompt → generate a response → get reward from RM → episode ends.
- **Key mechanism**: KL penalty. At each token, penalize the policy for diverging from the SFT model:
  ```
  reward_total = r_θ(x, y) − β · log(π_RL(y|x) / π_SFT(y|x))
  ```
- **PPO update**: Compute advantages using the RM as value function. Clip policy ratio to [1−ε, 1+ε]. Optimize the clipped surrogate objective.
- Training: 50 PPO steps, batch size 16, β=0.1, ε=0.2, lr=1e-5.
- **Output**: `ppo_model` — the final aligned model.

### 7. Before/After Comparison
- Generate responses from the base (untrained) model, the SFT model, and the PPO model on a set of test prompts.
- Print side-by-side comparisons to show the improvement in instruction-following quality.
- Plot reward model scores across the three model versions to show the RLHF improvement.
- Plot training curves: SFT loss, RM accuracy, PPO reward over time.

## Key Functions/Classes

| Component | Class/Function | Purpose |
|---|---|---|
| Tokenizer | `SimpleTokenizer` | Character/word-level tokenizer for toy dataset |
| Language Model | `MiniGPT` | GPT decoder with causal attention, used as base for SFT/RM/PPO |
| SFT Training | `train_sft()` | Supervised fine-tuning loop with CE loss |
| Reward Model | `RewardModel` | SFT model + scalar head, trained on pairwise comparisons |
| RM Training | `train_reward_model()` | Pairwise ranking loss training loop |
| PPO Trainer | `PPOTrainer` | PPO optimization with KL penalty against SFT |
| Generation | `generate()` | Autoregressive sampling from any model |
| Comparison | `compare_outputs()` | Generate and display before/after responses |

## Data Flow

```
Toy Instructions → SFT Training → sft_model
                                      ↓
                    ┌─────────────────┴────────────────┐
                    ↓                                  ↓
              RewardModel                         PPO Policy (init from SFT)
              (scalar head on SFT)                KL penalty vs SFT
                    ↓                                  ↓
              RM Training                         PPO Training
              (pairwise loss)                     (clipped surrogate + KL)
                    ↓                                  ↓
              reward_model ──────────────────→ ppo_model (aligned)
                                                        ↓
                                         compare_outputs(base, sft, ppo)
```

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | This Notebook |
|---|---|---|
| Model size | GPT-3 (1.3B–175B) | MiniGPT (~1M params, d_model=128, 4 layers) |
| Tokenizer | BPE (50K vocab) | Character-level (~100 vocab) |
| Dataset | 13K–33K human-labeled prompts | ~200 synthetic instruction-response pairs |
| Human feedback | 40 trained labelers | Simulated heuristic ranking function |
| RM model size | 6B parameters | Same MiniGPT backbone + scalar head |
| PPO iterations | Multiple rounds (RM→PPO→collect→repeat) | Single round of PPO |
| KL coefficient β | Tuned per model size | Fixed at 0.1 |
| Pretraining mix (PPO-ptx) | Mixed pretraining gradients | Omitted (pure PPO) |
| Model sizes for RM | Up to 175B | Single small model |
| Evaluation | Human evaluations, TruthfulQA, toxicity | Reward model scores + qualitative comparison |

These simplifications keep the notebook runnable within Kaggle's 30-minute GPU limit while faithfully implementing the core RLHF algorithm (SFT → RM → PPO with KL penalty).
