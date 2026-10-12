# Language Models are Few-Shot Learners (GPT-3)

**arXiv:** https://arxiv.org/abs/2005.14165
**Authors:** Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei (OpenAI, 2020)

## Summary

GPT-3 demonstrated that scaling up autoregressive language models to 175 billion parameters enables strong task-agnostic, few-shot performance without any gradient updates or fine-tuning. Tasks and few-shot demonstrations are specified purely via text interaction with the model. GPT-3 achieves competitive results on translation, question-answering, cloze tasks, arithmetic, and novel-word usage, sometimes matching prior state-of-the-art fine-tuning approaches. The paper introduces three evaluation paradigms — **zero-shot** (task description only), **one-shot** (one example), and **few-shot** (K examples in the prompt) — and shows that performance consistently improves with model scale and number of in-context examples.

## Core Idea

The key insight is that large language models, trained on enough data, develop a *meta-learning* capability through in-context learning. Instead of updating weights for a new task, the model conditions on a prompt containing a task description and a few input-output examples, then generates the answer for a new input. This is essentially pattern completion: the model uses its pretrained knowledge to continue the pattern established by the examples.

The paper evaluates on dozens of NLP tasks: translation, QA, cloze, arithmetic, word unscrambling, novel word usage, news article generation, and SAT analogies. A central finding is the **scaling relationship** — both model size and number of in-context examples drive better performance, with the largest model (175B) showing the steepest improvement.

## Key Method Details Relevant to the Code

1. **In-context learning protocol:** The prompt is constructed as `task_description + K examples + query`, where each example is `input → output`. The model generates the output for the query via next-token prediction.
2. **Few-shot scaling:** Accuracy improves monotonically with K (0, 1, 2, 4, 8, 16, 32 examples).
3. **Model scaling:** Larger models use in-context examples more effectively — the gap between zero-shot and few-shot widens with scale.
4. **No gradient updates:** The model weights are frozen; all task knowledge is accessed through the prompt.

In our notebook, we replicate this protocol using a small pretrained language model (GPT-2) on a toy sentiment classification task. We construct prompts with 0, 1, 2, 4, 8, and 16 in-context examples and measure how classification accuracy scales with the number of examples.

## What Problem Does It Solve

Imagine you're learning a new board game. If someone just tells you the rules, that's like *zero-shot* — you might get it right, but you'll make mistakes. If they show you one example move, that's *one-shot* — a bit better. If they walk you through several rounds first, that's *few-shot* — you'll play much more confidently. Before GPT-3, AI systems needed hundreds or thousands of practice examples to learn a new task (like having to play 500 games before understanding the rules). GPT-3 showed that a big enough AI, having already read tons of text, can learn new tasks from just a handful of examples — the same way you pick up a new game after watching a few rounds. This paper proved that simply making language models *bigger* lets them learn this way, no special training required.

## Influence

GPT-3 popularized the in-context learning paradigm and prompt engineering, directly leading to GPT-3.5/4, ChatGPT, and the modern LLM era. It showed that scale alone can unlock emergent capabilities, influencing the entire trajectory of large language model development. The paper has been cited over 30,000 times and is one of the most influential AI papers of the decade.

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/language-models-few-shot-learners-gpt3)
