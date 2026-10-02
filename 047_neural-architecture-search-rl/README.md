# Neural Architecture Search with Reinforcement Learning

**Authors:** Barret Zoph, Quoc V. Le  
**Published:** November 2016 (ICLR 2017)  
**arXiv:** [1611.01578](https://arxiv.org/abs/1611.01578)  
**Citations:** 6,057 (620 influential) per Semantic Scholar  

[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/neural-architecture-search-rl)

## Summary

This paper introduced Neural Architecture Search (NAS) — using a recurrent neural network (RNN) as a "controller" that generates descriptions of neural network architectures, then training that controller with reinforcement learning (REINFORCE policy gradient) to maximize the validation accuracy of the architectures it produces. On CIFAR-10, the best architecture found by the controller achieved 3.65% test error, surpassing the previous state-of-the-art. On Penn Treebank, the method discovered a novel recurrent cell achieving 62.4 perplexity, outperforming LSTM. The controller is an LSTM that outputs hyperparameters layer-by-layer (filter height, filter width, stride, number of filters) for CNNs, or a tree-structured expression for RNN cells. After each child network is trained and evaluated, the validation accuracy serves as the reward signal. The controller is updated via REINFORCE with a moving average baseline to reduce variance.

## What problem does it solve?

Imagine you want to build the best possible Lego spaceship, but there are millions of ways to combine the pieces. Instead of trying every combination yourself (which would take forever), you build a tiny robot that designs spaceships for you. Every time the robot designs a spaceship that flies well, you tell it "good job!" and it learns to make more like that. When the spaceship doesn't fly well, it learns to avoid that design. Over time, the robot gets really good at designing spaceships — maybe even better than any human Lego builder.

That's exactly what this paper does, but with neural networks instead of Lego spaceships. Before this paper, designing a good neural network required human experts who spent months tweaking layer sizes, filter counts, and connections. Zoph and Le built an RNN "controller" that automatically designs neural networks and learns from each attempt's performance, eventually discovering architectures that match or beat human-designed ones.

## Key Method Details

- **Controller:** An LSTM that sequentially predicts hyperparameters for each layer of a CNN (filter height ∈ {1,3,5}, filter width ∈ {1,3,5}, stride ∈ {1,2,3}, number of filters ∈ {24,36,48,64})
- **Search space:** For CNNs, the controller generates architectures with a fixed number of layers (e.g., 6). Each layer has 5 hyperparameters. Skip connections are predicted via an anchor point mechanism.
- **Training child networks:** Each sampled architecture is trained from scratch on the target dataset (CIFAR-10). Validation accuracy is the reward.
- **REINFORCE update:** The controller is trained with policy gradient: ∇θ J(θ) ≈ (1/N) Σ (R_n − b) ∇θ log p(a_n|s_n), where R_n is the validation accuracy and b is an exponential moving average baseline.
- **Parallelism:** The paper uses asynchronous parameter server training to speed up the controller, and trains multiple child networks in parallel.
- **Simplifications in this notebook:** We use a much smaller search space (3 layers, fewer choices), MNIST instead of CIFAR-10, 1-2 epochs per child network, and synchronous (non-parallel) training.

## Influence

This paper pioneered the NAS field, directly inspiring NASNet (Zoph et al., 2018), ENAS (Pham et al., 2018), DARTS (Liu et al., 2019), and the entire AutoML industry. Google's AutoML products and Cloud AutoML trace their lineage to this work. It demonstrated that machine learning models could design their own architectures, fundamentally changing how neural networks are developed.
