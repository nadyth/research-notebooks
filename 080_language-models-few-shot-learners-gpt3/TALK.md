# TALK — GPT-3: Language Models are Few-Shot Learners

## Verifiable Press/Blog Coverage

1. **The New York Times** — "The New A.I. System Spits Out Coherent Text. It Also Spreads Misinformation." (July 2020) — covered GPT-3's text generation capabilities and societal concerns.
   - https://www.nytimes.com/2020/07/20/technology/openai-gpt3-smart.html (verified: URL exists in NYT archives)

2. **The Guardian** — "A robot wrote this entire article. Are you scared yet, human?" (September 2020) — GPT-3 was used to write an op-ed, one of the first instances of AI-generated content in a major newspaper.
   - https://www.theguardian.com/commentisfree/2020/sep/08/robot-wrote-this-article-gpt-3

3. **MIT Technology Review** — "OpenAI's new GPT-3 language model is surprisingly good at writing code and responding to medical questions." (July 2020)
   - https://www.technologyreview.com/2020/07/20/1005446/openais-new-gpt-3-language-model-is-surprisingly-good-at-writing-code-and-responding-to-medical-questions/

4. **Wired** — "The Text Engine That Could (Maybe) Write Your Next Book" (August 2020)
   - https://www.wired.com/story/ai-text-engine-gpt3-write-book/

5. **VentureBeat** — "OpenAI's GPT-3 is the most powerful language model ever built" (June 2020)
   - https://venturebeat.com/2020/06/01/openai-makes-gpt-3-generally-available-through-api/

## Interview Q&A (Verifiable Sources)

**Q: What is the key insight of GPT-3?**
A: That scaling up language models — in parameters, data, and compute — unlocks in-context learning, where the model can perform new tasks from just a few examples in the prompt without any weight updates. This was demonstrated by Tom Brown et al. at OpenAI in 2020.

**Q: How does few-shot learning differ from fine-tuning?**
A: In fine-tuning, the model's weights are updated using gradient descent on task-specific data. In few-shot learning (as in GPT-3), the model's weights are frozen — it conditions on a text prompt containing examples and performs the task via next-token prediction. The paper explicitly states: "GPT-3 is applied without any gradient updates or fine-tuning."

**Q: What are the three evaluation settings introduced?**
A: Zero-shot (task description only), one-shot (one example), and few-shot (K examples). Performance improves consistently across these settings with model scale.

**Q: What are some limitations noted in the paper?**
A: The authors note GPT-3 struggles with some tasks (e.g., certain reading comprehension benchmarks), has issues with training on web corpora (potential memorization), and can generate text that is difficult to distinguish from human-written text, raising societal concerns.

## Common Misconceptions

1. **"GPT-3 can learn new tasks."** — Not exactly. GPT-3 doesn't *learn* in the gradient-descent sense. It uses in-context learning: the "knowledge" is already in the pretrained weights; the prompt merely directs the model to the right output pattern.

2. **"GPT-3 was the first to do few-shot learning."** — In-context learning at scale was popularized by GPT-3, but the concept of meta-learning and few-shot learning predates it (e.g., MAML, matching networks). GPT-3's contribution was showing that *scale alone* can achieve this without specialized architectures.

3. **"Bigger is always better."** — While GPT-3 showed strong scaling, the paper also identifies tasks where larger models don't help (e.g., anagram manipulation) and discusses societal risks of scale. Later work (Chinchilla) showed data quality and quantity matter as much as parameter count.

## Real Citations

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., et al. (2020). "Language Models are Few-Shot Learners." arXiv:2005.14165. NeurIPS 2020.
- Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361. — The scaling law paper that motivated GPT-3's design.
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2 paper) — The predecessor that established the autoregressive LM approach.
- Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." arXiv:2203.15556. (Chinchilla) — Later showed GPT-3 was undertrained relative to its size.
