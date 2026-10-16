# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**arXiv:** [2203.02155](https://arxiv.org/abs/2203.02155) | **Authors:** Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe (OpenAI) | **Year:** 2022

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/instructgpt-human-feedback)

## Summary

InstructGPT is OpenAI's landmark paper showing that language models can be aligned with human intent through a three-step process called RLHF (Reinforcement Learning from Human Feedback). Rather than simply making models larger, the authors fine-tune GPT-3 using human feedback to make it better at following instructions — more helpful, truthful, and harmless. The process works as follows: (1) collect human-written demonstrations of ideal responses and fine-tune the model with supervised learning (SFT); (2) collect human rankings of model outputs and train a reward model (RM) to predict which response a human would prefer; (3) use the reward model as a reward signal to further fine-tune the SFT model with PPO (Proximal Policy Optimization). The stunning result: a 1.3B-parameter InstructGPT model produces outputs preferred over the 175B GPT-3 — 100x smaller but better aligned. This paper directly led to ChatGPT and established RLHF as the standard technique for aligning large language models.

## Core Idea

The key insight is that the language modeling objective (predict the next token on internet text) is **misaligned** with the user's actual goal (follow instructions helpfully and safely). Simply scaling up model size doesn't fix this — bigger models still generate untruthful, toxic, or unhelpful content. RLHF fixes the alignment by using human preferences as a reward signal:

1. **Supervised Fine-Tuning (SFT):** Human labelers write high-quality responses to prompts. GPT-3 is fine-tuned on these demonstrations with standard supervised learning (cross-entropy loss, 16 epochs, cosine LR decay, residual dropout 0.2).

2. **Reward Modeling (RM):** Starting from the SFT model with the unembedding layer removed, a reward model is trained to output a scalar reward for (prompt, response) pairs. Labelers rank K=4 to K=9 responses for each prompt. The RM is trained with a cross-entropy loss where the reward difference represents the log-odds of one response being preferred over another:

   > loss(θ) = −1/C(K,2) · E[log(σ(r_θ(x, y_w) − r_θ(x, y_l)))]

   where y_w is the preferred (winning) response and y_l is the losing response. All C(K,2) comparisons from one prompt are batched together to prevent overfitting.

3. **PPO Optimization:** The SFT model is fine-tuned against the reward model using PPO in a bandit environment. A per-token KL penalty from the SFT model prevents the policy from drifting too far:

   > objective(φ) = E[r_θ(x,y) − β·log(π_φ(y|x)/π_SFT(y|x))] + γ·E[log(π_φ(x))]

   The KL coefficient β controls penalty strength; γ controls pretraining mix (PPO-ptx). Steps 2 and 3 can be iterated: collect more comparisons from the current best policy → retrain RM → retrain policy.

## Key Method Details

- **Model sizes:** SFT, RM, and PPO models were trained at 1.3B, 6B, and 175B parameter scales. The best results came from 175B PPO, but even 1.3B InstructGPT outperformed 175B GPT-3.
- **Reward model:** Only 6B RMs were used in practice — 175B RMs were unstable and too expensive as value functions during RL.
- **KL penalty:** Prevents the RL policy from over-optimizing the reward model (which can exploit reward model blind spots). The value function is initialized from the RM.
- **PPO-ptx:** Mixing pretraining gradients (γ > 0) fixes performance regressions on public NLP benchmarks (NLI, QA, etc.) that pure PPO can cause.
- **Labeler agreement:** Training labelers agreed 72.6 ± 1.5% of the time; held-out labelers agreed 77.3 ± 1.3%.
- **Dataset:** ~13K SFT prompts, ~33K RM prompts, ~31K PPO prompts. Use-case distribution: 45.6% generation, 12.4% open QA, 11.2% brainstorming, 8.4% chat, etc.

## What problem does it solve

Imagine you have a super-smart robot that has read every book and website in the world. You ask it "Write me a nice bedtime story about a bunny." Instead of a nice story, it might give you something weird — maybe it copies the style of a scary news article, or says something mean, or just rambles about rabbits in a way that isn't helpful at all. The robot knows *about* stories, but it doesn't know *how to be helpful* when you ask it to do something.

This paper solves that problem. The researchers realized that just making the robot bigger (giving it more books to read) doesn't make it better at following instructions. Instead, they hired real people to show the robot what a *good* response looks like, then had other people rank which robot responses were better or worse. They used those rankings to teach the robot a "reward system" — a way to score how good its answers are — and then trained the robot to always try to get the highest score. The amazing result: a much smaller robot trained this way gave better answers than a giant robot that was 100 times bigger but never learned to be helpful. This is exactly how ChatGPT was built!

## Influence

InstructGPT is one of the most consequential papers in AI history. It introduced the RLHF pipeline that became the foundation for ChatGPT (launched in November 2022), which triggered the current AI boom. The three-step SFT → RM → PPO recipe is now standard practice at every major AI lab (Anthropic, Google, Meta, Mistral). The paper demonstrated that alignment through human feedback is more efficient than model scaling — a paradigm shift that redirected the field from "just scale up" to "align carefully." It also crystallized the helpful/honest/harmless framework for evaluating AI safety.
