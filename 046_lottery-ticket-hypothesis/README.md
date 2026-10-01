# The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks

**Paper:** [arXiv:1803.03635](https://arxiv.org/abs/1803.03635) | **Authors:** Jonathan Frankle, Michael Carbin | **Venue:** ICLR 2019 (Best Paper) | **Year:** 2018

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/lottery-ticket-hypothesis)

## Summary

The Lottery Ticket Hypothesis reveals that within a dense, randomly-initialized neural network, there exist sparse subnetworks ("winning tickets") that—when trained in isolation from their original initialization—can match the full network's test accuracy in a similar number of iterations. The key insight is that it's not just the architecture that matters, but the specific *initial weights* of the surviving connections. These winning tickets "won the initialization lottery": their initial weights were already well-suited for effective training. The paper introduces **iterative magnitude pruning (IMP)**—train, prune the smallest-magnitude weights, reset remaining weights to their original initialization, and repeat—to discover these subnetworks, consistently finding tickets at 10–20% of the original network size that train faster and reach higher accuracy than the dense baseline.

## What Problem Does It Solve?

Imagine you have a giant box of 1,000 Lego pieces, but you only actually need about 150 of them to build your castle. The rest are just sitting there taking up space and making everything slower. But here's the catch: you don't know *which* 150 pieces are the right ones until you've already built the whole castle once.

That's the problem this paper solves. Before this research, people knew you could shrink a trained neural network by removing unimportant connections (pruning), making it smaller and faster. But if you tried to train that small network from scratch, it would fail badly—it couldn't learn at all. It was like giving someone the right 150 Lego pieces but with no instructions.

The authors discovered something surprising: if you keep the *original starting weights* of those 150 important pieces (instead of giving them new random starting weights), the small network trains just as well as the big one! It's like those 150 pieces came with a head start—they were "lucky" from the beginning. They called these lucky small networks "winning lottery tickets."

This matters because it means we can build AI systems that are much smaller, faster, and cheaper to run, without sacrificing performance—which is crucial for deploying AI on phones, in cars, or anywhere with limited computing power.

## Core Idea & Key Method Details

- **Iterative Magnitude Pruning (IMP):** The algorithm iteratively (1) trains the network to convergence, (2) prunes a fraction (e.g., 20%) of the weights with the smallest absolute magnitude, (3) resets the surviving weights to their *original* initialization values, and (4) repeats from step 1 on the pruned subnetwork.
- **Winning Ticket:** A subnetwork (mask + original initialization) that, when trained in isolation, matches the full network's accuracy. The mask is the binary pruning pattern; the initialization is the original random weights at those positions.
- **Key Finding—Initialization Matters:** If the same pruned architecture is re-initialized with *new* random weights (instead of the original ones), it trains poorly. This proves that the specific initial weight values, not just the architecture, are crucial.
- **Pruning Rate:** Typically 20% per iteration (P=20%), repeated N times to reach target sparsity. The paper uses P=51% for iterative pruning on deeper networks.
- **Early Stopping / Rewinding:** For deeper networks, weights are reset to their value at iteration *k* (not iteration 0) to handle optimization difficulties—a technique called "weight rewinding."

## Influence

With over 1,300 citations, this paper fundamentally changed how the community thinks about pruning and sparse networks. It launched an entire research direction into *when* and *why* sparse subnetworks exist, led to the discovery of lottery tickets in transformers (BERT), reinforcement learning, and even proved theoretical guarantees (Malach et al., 2019). It was awarded the ICLR 2019 Best Paper Award.

## Notebook

The notebook implements iterative magnitude pruning from scratch on a small MLP trained on MNIST, extracts winning tickets at various sparsity levels, and compares them against re-initialized subnetworks to demonstrate the core finding of the paper.
