# DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

**Paper:** [arXiv:1910.01108](https://arxiv.org/abs/1910.01108)
**Authors:** Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf (HuggingFace)
**Published:** October 2019 (last revised March 2020)

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/078-distilbert-distilled-version-of-bert)

## Summary

DistilBERT is a compressed version of BERT that retains 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster for inference. The key innovation is applying knowledge distillation during the **pre-training phase** rather than only at the task-specific fine-tuning stage (which is what most prior distillation work did). The student model is a 6-layer Transformer encoder (half of BERT-base's 12 layers), initialized from the teacher's layers, and trained with a **triple loss**: (1) a distillation loss (KL divergence between the student's and teacher's softened output distributions), (2) a masked language modeling (MLM) loss (the student's own MLM objective), and (3) a cosine embedding loss (aligning the student's hidden states with the teacher's). The token type embeddings and the pooler are removed to further reduce parameters. DistilBERT has ~66M parameters vs BERT-base's ~110M.

## What problem does it solve

Imagine you have a really smart teacher who knows everything about English — but the teacher is very big and slow, so it's hard to put them in your phone or a small device. DistilBERT is like training a smaller, younger "student" assistant who watches what the big teacher does and learns to copy most of their knowledge. The student is much smaller, so you can carry them in your phone, and they answer much faster. The cool trick is that the student doesn't just learn from textbooks — they also learn by watching *how* the teacher thinks about problems (what answers the teacher is almost-certain about vs unsure about), which helps the student learn better than if they studied alone. The student ends up almost as good as the teacher, but takes up much less space and works much faster.

## Core idea and method details

- **Knowledge distillation at pre-training time**: Instead of distilling only during fine-tuning, DistilBERT distills the pre-trained model itself. The teacher is BERT-base (12 layers, 110M params); the student is a 6-layer Transformer.
- **Architecture**: The student has 6 Transformer layers, 768 hidden size, 12 attention heads — same hidden dimension as BERT-base but half the layers. Token type embeddings and the pooler are removed.
- **Initialization**: The student's 6 layers are initialized by copying every other layer from BERT-base (layers 0, 2, 4, 7, 9, 11).
- **Triple loss**:
  1. **Distillation loss**: KL divergence between softmax-temperature-scaled output distributions of student and teacher. Temperature T=2 is used; the loss is T² · KL(softmax(teacher_logits/T) || softmax(student_logits/T)).
  2. **MLM loss**: Standard masked language modeling cross-entropy loss (same as BERT's pre-training objective).
  3. **Cosine embedding loss**: 1 − cos(student_hidden, teacher_hidden) on the final hidden states, aligning the student's representations with the teacher's.
- **Training**: DistilBERT was trained on the same data as BERT (English Wikipedia + Toronto BookCorpus) with dynamic masking. The distillation process took ~3.5 days on 8 V100 GPUs — significantly cheaper than BERT's original pre-training.
- **Results**: On the GLUE benchmark, DistilBERT retains ~97% of BERT-base's performance. On the downstream IMDB sentiment classification task, it achieves comparable accuracy with 60% faster inference.

## Influence

DistilBERT is one of the most widely used compressed language models in production. It is a core part of the HuggingFace `transformers` library and has been downloaded hundreds of millions of times. It demonstrated that pre-training distillation is viable and inspired a line of work on efficient transformer variants (TinyBERT, MobileBERT, etc.). The model is commonly deployed for on-device and edge NLP applications where latency and memory matter.
