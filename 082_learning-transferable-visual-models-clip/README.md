# Learning Transferable Visual Models From Natural Language Supervision (CLIP)

**arXiv:** https://arxiv.org/abs/2103.00020

**Authors:** Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever (OpenAI)

**Year:** 2021

## Summary

CLIP (Contrastive Language–Image Pre-training) introduces a radically simple yet powerful approach to learning visual representations: instead of training a classifier on a fixed set of categories, CLIP trains a dual encoder — an image encoder and a text encoder — to align matching image–caption pairs in a shared embedding space using a contrastive loss (InfoNCE). Trained on 400 million (image, text) pairs scraped from the internet, CLIP achieves zero-shot transfer to over 30 downstream benchmarks, matching ResNet-50's ImageNet accuracy without seeing a single ImageNet training example. After pre-training, classification becomes a text-prompt problem: encode the image and a set of text prompts like "a photo of a dog", "a photo of a cat", and pick the nearest text embedding. This approach bridges natural language and vision in a way that generalizes to arbitrary visual concepts described in words.

## Core Idea

The key insight is that natural language provides a much richer supervision signal than fixed class labels. A caption like "a photo of my golden retriever catching a frisbee at the beach" contains far more information than the label "dog". CLIP exploits this by learning to match images with their corresponding text descriptions in a shared 512-dimensional (or 256/64 in smaller variants) embedding space.

**Contrastive Learning:** Given a batch of N (image, text) pairs, CLIP computes an N×N matrix of cosine similarities between all image and text embeddings. The loss maximizes the diagonal (matching pairs) while minimizing off-diagonal (mismatched pairs) using symmetric cross-entropy — essentially a symmetric InfoNCE loss.

**Architecture:**
- **Image encoder:** ResNet-50/101 or Vision Transformer (ViT-B/32, ViT-B/16, ViT-L/14)
- **Text encoder:** Transformer with 12 layers, 512 width, 8 heads (for the base model)
- Both encoders project their outputs to a shared embedding dimension via a linear projection head

**Zero-shot Classification:** To classify an image, encode it alongside text prompts for each class (e.g., "a photo of a {class}"). The predicted class is the one whose text embedding has the highest cosine similarity to the image embedding.

## Key Method Details Relevant to the Code

1. **Dual encoder architecture** — separate image and text encoders that map to a shared embedding space
2. **Learnable temperature** — a log-scale temperature parameter τ controls the sharpness of the softmax in the contrastive loss
3. **Symmetric contrastive loss** — cross-entropy applied in both directions (image→text and text→image)
4. **Prompt engineering** — zero-shot performance depends on prompt templates ("a photo of a {}")
5. **Large-scale training** — the paper uses 400M pairs; our toy implementation uses a small synthetic dataset

## What Problem Does It Solve

Imagine you have a robot that can recognize cats, dogs, and cars because you showed it thousands of labeled pictures. But now you want it to recognize elephants — and you'd have to go collect thousands of elephant photos, label them, and retrain the whole robot. That's how old AI worked: every new thing meant a lot of new labeled data and retraining.

CLIP solves this by teaching the robot to understand *language* alongside *pictures*. Instead of just learning "this is a cat," it learns "this picture matches the sentence 'a fluffy orange cat sitting on a windowsill.'" So when you want the robot to recognize elephants, you just *tell* it in words: "Show me pictures of an elephant." No new training needed! It's like teaching someone to read a dictionary instead of memorizing flashcards — once they can connect words to images, they can recognize anything you can describe.

## Influence

CLIP became one of the most influential papers in multimodal AI:
- **Foundation for generative models:** DALL-E, Stable Diffusion, and Midjourney all use CLIP's text encoder to guide image generation
- **Open-vocabulary detection & segmentation:** GroundingDINO, SAM (Segment Anything) leverage CLIP-like alignment
- **CLIP embeddings as a universal signal:** used in retrieval, classification, clustering, and even robotics
- **Sparked the multimodal foundation model era:** Flamingo, BLIP, LLaVA, and GPT-4V all build on CLIP's vision-language alignment paradigm
- **OpenAI released model weights** publicly, enabling widespread adoption

**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/082_learning-transferable-visual-models-clip)
