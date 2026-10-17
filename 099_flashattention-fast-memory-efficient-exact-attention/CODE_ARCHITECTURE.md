# Code Architecture — FlashAttention Notebook

## Overview

The notebook implements a simplified version of the FlashAttention algorithm in PyTorch, demonstrating the core tiling and online softmax techniques. It benchmarks the tiled implementation against standard (naive) attention in terms of memory usage and wall-clock speed at increasing sequence lengths.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Import torch, torch.nn.functional as F, time, math, matplotlib.
- Set random seeds for reproducibility.
- Define device (CPU for this notebook; the algorithm is device-agnostic in principle).

### 2. Naive Attention (Baseline)
- **Function:** `naive_attention(Q, K, V)` — computes S = QK^T, P = softmax(S), O = PV.
- This materializes the full N×N attention matrix, demonstrating the O(N²) memory bottleneck.
- Includes causal mask option.
- **Shapes:** Q, K, V: (batch, heads, N, d) → O: (batch, heads, N, d)

### 3. FlashAttention Forward (Tiled)
- **Function:** `flash_attention_forward(Q, K, V, block_size)` — the core algorithm.
- **Key components:**
  - **Outer loop** over blocks of K, V (columns of the attention matrix).
  - **Inner loop** over blocks of Q (rows of the attention matrix).
  - **Online softmax tracking:** For each Q block, maintain running max `m` and running sum `l`.
  - **Rescaling:** When processing a new K block, update `m` to max of old and new block max, rescale the running output `O` by `exp(old_m - new_m)`, and accumulate.
- **Algorithm (simplified FlashAttention forward):**
  ```
  Initialize O = 0, m = -inf, l = 0 for each row
  For each block j of K, V:
      Load Kj, Vj to SRAM (i.e., slice the tensors)
      For each block i of Q:
          Qi = Q[i_block]
          Sij = Qi @ Kj.T
          mij = rowmax(Sij)
          Pij = exp(Sij - mij)
          lij = rowsum(Pij)
          m_new = max(m_old, mij)
          l_new = exp(m_old - m_new) * l_old + exp(mij - m_new) * lij
          O_new = (exp(m_old - m_new) * l_old * O_old + exp(mij - m_new) * Pij @ Vj) / l_new
          Update m, l, O
  Return O
  ```
- **Shapes:** Same as naive: (batch, heads, N, d) → (batch, heads, N, d)
- **Deliberate simplifications vs. paper:**
  - No CUDA kernel fusion — uses PyTorch ops on CPU/GPU tensors.
  - No backward pass recompute — forward-only demonstration.
  - No dropout or masking in the tiled version (causal mask can be added as extension).
  - Block size is a tunable parameter rather than hardware-derived.

### 4. Correctness Verification
- Compare FlashAttention output against `F.scaled_dot_product_attention` and naive attention.
- Assert max absolute difference < 1e-5 (tolerance for floating-point accumulation differences).
- Test with multiple head dimensions (d=64, d=128) and sequence lengths.

### 5. Memory Benchmark
- Measure peak memory for both implementations at increasing sequence lengths.
- Use `torch.cuda.max_memory_allocated()` if GPU available, otherwise track tensor sizes manually.
- Naive: O(N²) per head (the S and P matrices).
- Flash: O(Nd) per head (only Q, K, V, O stored; blocks are small).
- Plot memory vs. sequence length showing the divergence.

### 6. Speed Benchmark
- Time both implementations at N = 256, 512, 1024, 2048, 4096.
- Use `time.perf_counter()` with warmup runs.
- Plot wall-clock time vs. sequence length.
- Note: on CPU, the tiled version may be slower than naive (no kernel fusion benefit), but it uses less memory. On GPU, FlashAttention would be faster due to reduced HBM access. The notebook demonstrates the memory advantage clearly.

### 7. Analysis & Visualization
- Plot: memory usage vs. sequence length (naive vs. Flash).
- Plot: wall-clock time vs. sequence length (naive vs. Flash).
- Print a table summarizing speedup and memory savings at each sequence length.
- Discuss the trade-off: FlashAttention trades extra FLOPs (recomputation) for fewer memory accesses.

### 8. (Bonus) Causal Masking Extension
- Show how to incorporate causal masking in the tiled version by skipping blocks above the diagonal.
- This demonstrates the block-sparse extension concept from the paper.

## Data Flow

```
Input: Q, K, V ∈ R^{B×H×N×d}
  │
  ├── Naive: S = QK^T → softmax → PV   [materializes N×N matrix]
  │
  └── Flash: for each (Qi, Kj, Vj) block:
              Sij = Qi @ Kj.T
              update running (m, l, O)
              never store full N×N         [O(Nd) memory]
  │
Output: O ∈ R^{B×H×N×d}  (identical for both methods)
```

## Key Functions/Classes

| Function | Purpose | Input → Output |
|---|---|---|
| `naive_attention(Q, K, V)` | Standard attention baseline | (B,H,N,d)×3 → (B,H,N,d) |
| `flash_attention_forward(Q, K, V, block_size)` | Tiled attention with online softmax | (B,H,N,d)×3 → (B,H,N,d) |
| `measure_memory(fn, *args)` | Memory profiler for a function | returns peak memory in bytes |
| `benchmark(fn, *args, warmup, repeats)` | Timing wrapper | returns median time in seconds |
| `plot_benchmarks(results)` | Visualization helper | plots memory & speed curves |

## Deliberate Simplifications vs. Full Paper

1. **No CUDA kernel:** The real FlashAttention is a fused CUDA kernel. This notebook uses PyTorch ops for clarity.
2. **Forward only:** No backward pass / recomputation of gradients. The paper's key insight for the backward pass is mentioned in markdown but not implemented.
3. **No block-sparse attention:** The sparse extension is discussed but not benchmarked.
4. **CPU benchmarking:** On CPU, the memory advantage is real but the speed advantage may not materialize (no HBM/SRAM distinction). The notebook notes this clearly.
5. **Small dimensions:** Head dimension d=64, batch=1, heads=1 for clarity. Real usage has d up to 256 and many heads.
