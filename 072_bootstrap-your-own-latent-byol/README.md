# Bootstrap Your Own Latent (BYOL) — A New Approach to Self-Supervised Learning

**arXiv:** [2006.07733](https://arxiv.org/abs/2006.07733)  
**Authors:** Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko (DeepMind)  
**Published:** June 2020, NeurIPS 2020  
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/bootstrap-your-own-latent-byol)

## Summary

BYOL (Bootstrap Your Own Latent) is a self-supervised image representation learning method that learns meaningful visual features *without using negative pairs*. This was a radical departure from dominant contrastive methods like SimCLR and MoCo, which all relied on comparing an anchor image against many "negative" images to avoid trivial solutions. BYOL instead uses two neural networks — an **online** network and a **target** network — that interact and learn from each other. The online network is trained to predict the target network's representation of the same image under a different augmented view. Simultaneously, the target network is updated as a slow-moving average (EMA) of the online network's parameters. The key architectural ingredient is a **predictor head** in the online network, which, combined with the stop-gradient on the target, prevents representational collapse.

BYOL achieves 74.3% top-1 linear evaluation accuracy on ImageNet with a ResNet-50, and 79.6% with a larger architecture — both surpassing SimCLR at the time, without needing negative pairs or large batch sizes.

## Key Method Details

- **Online network** (weights θ): encoder f_θ → projector g_θ → predictor q_θ. Three stages.
- **Target network** (weights ξ): encoder f_ξ → projector g_ξ. Same architecture as online, but no predictor. Updated only via EMA: ξ ← τ·ξ + (1−τ)·θ, where τ follows a cosine schedule from 0.996 to 1.0.
- **No negative pairs**: BYOL's loss is a simple mean-squared error between the L2-normalized online prediction and the L2-normalized target projection — no contrastive comparison with other images.
- **Stop-gradient**: gradients do not flow through the target network; the target output is treated as a fixed regression target.
- **Loss function**: L = 2 − 2·⟨normalize(q_θ(g_θ(f_θ(v))), normalize(g_ξ(f_ξ(v′)))⟩, which is equivalent to the squared L2 distance between normalized prediction and target.
- **Augmentation**: two views v and v′ from the same image using random crop, color jitter, horizontal flip, grayscale, Gaussian blur, and solarization. BYOL's augmentation recipe is asymmetric — the first view gets stronger augmentation.
- **BatchNorm in projector/predictor**: critical for preventing collapse. Ablations show that removing BatchNorm from the MLP heads causes BYOL to collapse without negative pairs.
- **Optimizer**: LARS with cosine learning rate decay, base LR 0.2 scaled linearly with batch size, weight decay 1.5×10⁻⁶, 1000 epochs with 10-epoch warmup.
- **Evaluation**: linear probing — freeze encoder, train a linear classifier on top.

## What Problem Does It Solve

Imagine you're learning to recognize different types of fruit, but nobody ever tells you which fruit is which — you just have a big pile of photos. Previous methods solved this by showing the computer two photos at a time and saying "these are different fruits, learn to tell them apart." But this requires carefully picking which photos to compare, and if you pick bad comparisons, the computer learns nothing useful.

BYOL takes a completely different approach. It's like teaching yourself to draw: you look at a fruit from one angle, try to draw it, then look at the same fruit from a different angle and check if your drawing still makes sense. You're not comparing against other fruits — you're just trying to be consistent with yourself. The computer does the same: it takes one view of an image, tries to predict what a slowly-evolving "teacher" version of itself would say about a different view of the same image. The teacher is just a delayed copy of the student, so the student is essentially bootstrapping itself — pulling itself up by its own bootstraps, hence the name.

The surprising discovery is that this works *without* comparing different images at all. The predictor head and the slow-moving teacher prevent the system from cheating (just outputting the same thing for every image, which would be the easy way out). This is a big deal because it means you don't need huge batches of images to find good negatives — making the method simpler and more efficient than contrastive approaches.

## Influence

BYOL was a landmark paper that challenged the prevailing assumption that negative pairs are necessary for self-supervised learning. Published at NeurIPS 2020, it sparked significant debate (particularly around the role of BatchNorm) and inspired follow-up methods like SimSiam (which showed that stop-gradient + predictor suffices even without EMA), Barlow Twins, and VicReg. BYOL is widely cited (over 4,000 citations) and is considered one of the three pillars of modern self-supervised visual learning alongside SimCLR and MoCo. The paper's insight that predictor + stop-gradient prevents collapse opened an entire research direction in non-contrastive self-supervised learning.
