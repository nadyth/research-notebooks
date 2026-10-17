# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/flashattention-io-aware-exact-attention)


**Paper:** [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)  
**Authors:** Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré  
**Venue:** NeurIPS 2022  
**Code:** [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)  
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/flashattention-io-aware-exact-attention)

## Summary

FlashAttention is an IO-aware exact attention algorithm that uses tiling to reduce memory reads/writes between GPU high bandwidth memory (HBM) and on-chip SRAM. Standard attention materializes the full N×N attention matrix in HBM, making it slow and memory-hungry for long sequences. FlashAttention avoids this by computing attention in blocks: it loads blocks of Q, K, V into fast SRAM, computes partial softmax statistics incrementally, and writes only the final output back to HBM. The key insight is that softmax can be decomposed across blocks by tracking running maximum and sum statistics, allowing exact computation without ever materializing the full attention matrix. For the backward pass, FlashAttention recomputes the attention matrix on-chip from the stored softmax statistics rather than reading it back from HBM — trading extra FLOPs for dramatically fewer memory accesses. The algorithm requires O(N²d²M⁻¹) HBM accesses (where M is SRAM size) compared to Ω(Nd + N²) for standard attention, and is provably optimal across a range of SRAM sizes. FlashAttention yields 2-4× wall-clock speedup on GPT-2 and long-range arena, enables training Transformers with sequences up to 64K, and was the first to achieve better-than-chance performance on the Path-X and Path-256 benchmarks.

## What problem does it solve?

Imagine you're reading a very long book and trying to remember how every sentence relates to every other sentence. If the book has 1,000 sentences, you'd need to think about 1,000 × 1,000 = 1,000,000 relationships. That's what a Transformer AI does with "attention" — it compares every word to every other word. For short texts this is fine, but for long texts (like a whole book or a long conversation), the computer runs out of memory and gets really slow.

The clever trick FlashAttention uses is like this: instead of spreading out the entire book on your desk all at once (which would require a huge desk), you read it one chapter at a time. You keep a small notepad (the GPU's fast but tiny memory) where you write down key summaries as you go. When you finish all chapters, you have the complete answer — but you never needed the giant desk. This means AI can now "read" much longer texts — 16,000 or even 64,000 words — without running out of memory, making it possible for chatbots to remember long conversations and for AI to understand entire documents at once.

## Core Idea

1. **Tiling:** Split Q, K, V into blocks. Load one block at a time from slow HBM to fast SRAM. Compute partial attention within each block.
2. **Online softmax:** Track running maximum (m) and running sum (ℓ) statistics across blocks to correctly normalize the softmax without seeing the full matrix.
3. **Recomputation for backward:** Instead of storing the N×N attention matrix for backprop, recompute it on-chip from stored (m, ℓ) statistics — faster than reading from HBM.
4. **Kernel fusion:** Fuse QK^T, softmax, and PV into a single GPU kernel to eliminate intermediate HBM writes.

## Key Method Details

- **IO complexity:** O(N²d²/M) HBM accesses vs Ω(Nd + N²) for standard attention, where M is SRAM size and d is head dimension.
- **Memory:** O(Nd) instead of O(N²) — linear in sequence length.
- **Block-sparse extension:** Apply a sparsity mask to the blocks for further 2-4× speedup, scaling to 64K sequences.
- **Exact computation:** No approximation — FlashAttention computes the same result as standard attention, just with fewer memory accesses.

## Influence

FlashAttention has become one of the most widely adopted innovations in modern deep learning. The official repository has 25,000+ GitHub stars. It is integrated into PyTorch 2.0+ (as `scaled_dot_product_attention`), HuggingFace Transformers, Megatron-LM, and is used in training virtually all modern large language models including GPT-4, LLaMA, and Mistral. FlashAttention-2 (2023) improved parallelism and work partitioning, and FlashAttention-3 (2024) optimized for Hopper GPUs with FP8 support. The paper was featured in IEEE Spectrum and has been cited 5,000+ times.

---

## Resources

**Kaggle Notebook:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/flashattention-io-aware-exact-attention)

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, interview Q&A, citations, and common misconceptions about the paper |
