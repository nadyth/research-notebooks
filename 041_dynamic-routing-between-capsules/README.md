# Dynamic Routing Between Capsules

**Authors:** Sara Sabour, Nicholas Frosst, Geoffrey E. Hinton (Google Brain, Toronto)
**arXiv:** [1710.09829](https://arxiv.org/abs/1710.09829) — Submitted 26 Oct 2017

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/dynamic-routing-between-capsules)

## Summary

This paper introduces **Capsule Networks (CapsNet)**, a novel architecture that replaces scalar-output neurons of traditional CNNs with vector-output capsules. Each capsule is a group of neurons whose activity vector represents the *instantiation parameters* of a specific entity (an object or an object part). The **length** of the activity vector encodes the probability that the entity exists, while the **orientation** encodes its properties (pose, scale, thickness, etc.). Instead of max-pooling — which throws away positional information — CapsNets use an iterative **routing-by-agreement** mechanism: lower-level capsules predict the outputs of higher-level capsules via learned transformation matrices, and send their output to whichever higher-level capsule their prediction best agrees with. The paper demonstrates state-of-the-art MNIST accuracy (0.25% test error with a 3-layer network) and superior robustness to affine transformations, as well as the ability to segment highly overlapping digits — a task where traditional CNNs struggle.

### Core Idea

The central insight is that **CNNs lose spatial relationships** through max-pooling. A capsule preserves these relationships by outputting a *vector* rather than a scalar, where the vector's direction encodes the entity's pose. The dynamic routing algorithm iteratively refines which lower-level capsules "vote for" which higher-level capsules by measuring agreement (scalar product) between predictions and actual outputs. This replaces the static, crude routing of max-pooling with a learned, input-dependent routing mechanism.

### Key Method Details

- **Squashing function:** Non-linearity that preserves vector orientation but scales magnitude to [0, 1]: `v_j = (||s_j||² / (1 + ||s_j||²)) * (s_j / ||s_j||)`
- **Routing-by-agreement (Procedure 1):** Coupling coefficients `c_ij` are computed via softmax over logits `b_ij`. Predictions `û_j|i = W_ij * u_i` are weighted by `c_ij` to form `s_j`. Agreement `a_ij = v_j · û_j|i` updates logits iteratively (typically 3 iterations).
- **Margin loss:** For each digit class k: `L_k = T_k * max(0, m+ - ||v_k||)² + λ(1 - T_k) * max(0, ||v_k|| - m-)²` with m+ = 0.9, m- = 0.1, λ = 0.5.
- **Architecture:** Conv1 (256, 9×9 kernels, stride 1, ReLU) → PrimaryCaps (32 channels of 8D capsules, 6×6 grid) → DigitCaps (10 capsules of 16D each, 3 routing iterations) → optional decoder (3 FC layers reconstructing the image).
- **Reconstruction regularizer:** Masks out all but the correct digit capsule's output, feeds to a decoder, minimizes Euclidean reconstruction loss. Encourages capsules to encode pose information.

### Influence

Capsule Networks sparked significant research into alternatives to CNNs and alternatives to max-pooling. While they did not replace CNNs in practice (due to computational cost and scaling difficulties), the ideas of routing-by-agreement, vector-based representations, and explicit pose encoding influenced subsequent work in equivariant neural networks, attention mechanisms, and set-based architectures. The paper has been cited thousands of times and remains a landmark in representation learning.

## What problem does it solve

Imagine you're looking at a picture of a **face**. A regular CNN looks at the picture and says: "I see an eye here, an eye there, a nose here, a mouth here — so it must be a face!" But it doesn't keep track of **where** each part is. So if you shuffle the parts around — put the eyes at the bottom and the mouth at the top — the CNN might still say "face!" because it only cares about whether the parts exist, not how they're arranged. That's because of a step called **max-pooling** that throws away positional information.

This paper fixes that by using **capsules** — tiny groups of neurons that output a *vector* (an arrow) instead of a single number. The arrow's **length** tells you "how sure" the capsule is that something exists, and the arrow's **direction** tells you *where and how* it's positioned (tilted, thick, thin, etc.). The capsules also talk to each other through a system called **routing-by-agreement**: lower-level capsules (that detect simple parts like eyes or nose-strokes) "vote" for higher-level capsules (that represent the whole face), and the votes go to whichever higher-level capsule the prediction matches best. It's like a team of detectives each making a guess about what the overall picture shows, and they all pool their evidence toward the detective whose theory best matches everyone's clues.

The result? A network that's better at understanding the *relationships between parts*, can recognize overlapping digits (like two numbers drawn on top of each other), and is more robust to rotations and shifts. It's like the difference between recognizing a face by just checking off a list of features versus understanding how those features fit together.
