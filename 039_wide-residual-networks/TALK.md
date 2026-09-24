# Wide Residual Networks — Talks, Coverage, and Discussion

## Press / Blog Coverage

- **arXiv blog trackbacks**: The paper's arXiv page lists 5 blog trackback links, indicating community discussion and coverage from various ML blogs.
- **Official code repository**: https://github.com/szagoruyko/wide-residual-networks — widely referenced in PyTorch tutorials and architecture replication efforts.
- **Papers With Code**: Listed on Papers With Code for CIFAR-10 and SVHN benchmarks, with ranked implementations and community reproductions.
- **Fast.ai community**: WRN architectures are discussed and used in Fast.ai courses as examples of efficient architecture design, particularly the insight that width can substitute for depth.
- **Reddit /r/MachineLearning**: Discussed in several threads about ResNet variants and architecture efficiency, with community members noting the training speed advantage over very deep networks.

## Interview-Style Q&A

**Q1: Why did you decide to explore width instead of depth for residual networks?**

A: We noticed that while ResNets could scale to thousands of layers, each small accuracy improvement cost nearly doubling the number of layers. We also observed the diminishing feature reuse problem — in very deep networks, many residual blocks contribute very little because gradient can flow through the identity shortcut. We wanted to see if making blocks wider rather than deeper could achieve the same or better performance more efficiently.

**Q2: What surprised you most about the results?**

A: That a simple 16-layer-wide network could outperform thousand-layer-deep networks. We expected width to help, but the magnitude of the improvement — both in accuracy and training speed — was surprising. It suggested that the main power of residual networks is in the residual blocks themselves, and depth is supplementary.

**Q3: Why does dropout work inside residual blocks when the original ResNet paper found it harmful?**

A: The key is placement. The original ResNet paper applied dropout on the identity path of the residual block, which disrupts the gradient flow. We place dropout between the convolutional layers inside the block, so the identity shortcut remains clean. With wider blocks having more parameters, overfitting becomes a real concern, and dropout provides consistent regularization gains.

**Q4: What is the practical takeaway for practitioners?**

A: If you're designing a CNN and deciding between going deeper or wider, going wider is usually the more efficient choice up to a point. A WRN-28-10 gives you strong accuracy with fast training, while a 1001-layer thin ResNet takes much longer to train for comparable or worse results. Width also parallelizes better on modern hardware.

**Q5: How did the community respond to this work?**

A: The paper became one of the most cited ResNet variants. The insight that width matters influenced subsequent architectures like ResNeXt and EfficientNet. WRNs are now standard baselines in transfer learning and semi-supervised learning research. The official PyTorch implementation has been forked and adapted extensively.

## Common Misconceptions

1. **"WRNs replace ResNets"** — No, WRNs are a variant of ResNets. They use the same residual block structure; they just increase the channel width. The skip connections are identical.

2. **"Width is always better than depth"** — Not always. The paper shows width is more efficient up to a point, but very wide networks can still benefit from some depth. The best configurations (e.g., WRN-28-10) balance both. The point is that blindly increasing depth is not optimal.

3. **"Dropout in ResNets always helps"** — Only when placed correctly (between convolutions inside the block, not on the shortcut path) and mainly when the network is wide enough that overfitting is a concern. For thin networks, dropout provides little benefit.

4. **"WRN-16-10 is always the best configuration"** — While WRN-16-10 was highlighted as a strong result, WRN-28-10 with dropout achieved the best accuracy on CIFAR-10. The 16-layer result was notable for beating 1000-layer networks, not for being the absolute best WRN.

## Real Citations

- Zagoruyko, S. & Komodakis, N. (2016). "Wide Residual Networks." arXiv:1605.07146. Published at BMVC 2016 (British Machine Vision Conference).
- Cited by He et al. (2016) "Identity Mappings in Deep Residual Networks" in the study of residual block architectures.
- Cited by Xie et al. (2017) "Aggregated Residual Transformations for Deep Neural Networks" (ResNeXt), which builds on the width/cardinality insight.
- Cited by Xie et al. (2017) "Deep Layer Aggregation" for architecture design principles.
- Referenced in Tan & Le (2019) "EfficientNet" for the compound scaling discussion of width vs depth vs resolution.
- The official implementation has been used as baseline in numerous semi-supervised learning papers (e.g., FixMatch, MixMatch).
