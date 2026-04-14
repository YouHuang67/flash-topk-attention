English | [中文](README_CN.md)

# Flash TopK Attention

A fused kernel library for Flash Attention with top-k KV block scoring and sparse attention.

- **V1** (`flash_topk_attn`): Single fused Triton kernel — scores KV blocks, selects top-k, and computes attention output in one pass.
- **V2** (`flash_topk_attn_v2`): Same API as V1, **2x–4x faster** — internally uses a modular Triton + CUDA pipeline.

---

## Algorithm

### Scoring

Standard Flash Attention computes $O = \text{softmax}(QK^\top / \sqrt{D})\ V$ using the online softmax trick, iterating over KV blocks without materializing the full $N \times N$ matrix. Here $B$ is batch size, $N$ is sequence length, $H$ is number of heads, and $D$ is head dimension.

This kernel additionally partitions the KV sequence into $M = N / b$ non-overlapping blocks of size $b$, and computes a **block-level attention score** for each block $j$:

$$s_j = \sum_{i \ \in\  \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}.$$

The $\text{top-}k$ block indices are then selected:

$$\mathbf{I} = \underset{j \in \{1,\ldots,M\}}{\text{argtop-}k}\ s_j,$$

The output $\mathbf{I}$ has shape $[B, H, N, k]$ and can drive a subsequent sparse attention pass with $O(k \cdot b)$ KV cost instead of $O(N)$.

### Sparse Attention

Given the top-k block indices $\mathbf{I}$ from scoring, sparse attention computes output over selected KV blocks only.

**Q-Block Shared Candidates**: Queries are grouped into blocks of size $g$ (`q_block_size`). For the $m$-th query block covering queries $[mg,\ (m+1)g)$, we construct a shared candidate set by taking the union of all queries' top-k indices:

$$\mathcal{C}_m = \bigcup_{t=mg}^{(m+1)g-1} \mathrm{TopK}(q_t)$$

The candidates are sorted in ascending order. Let $L_m = |\mathcal{C}_m| \leq g \cdot k$.

Each query $q_t$ in the block attends over the **entire** shared set $\mathcal{C}_m$:

$$O_t = \sum_{j \in \mathcal{C}_m} \frac{\exp(q_t k_j^\top / \sqrt{D})}{\sum_{j' \in \mathcal{C}_m} \exp(q_t k_{j'}^\top / \sqrt{D})} \cdot v_j$$

- When $g=1$: each query attends only to its own top-k blocks (per-query sparse attention)
- When $g>1$: queries share candidates, enabling batched KV access but slightly expanding each query's attention scope

### Virtual Padding

When N is not divisible by block size b, virtual padding extends the sequence to N' = pad_head + N + pad_tail where N' is divisible by b. The padding is purely logical: QKV data is unchanged, and padded positions are masked to negative infinity before softmax. For partial blocks at the head or tail, the block score is normalized by the number of valid tokens, ensuring fair top-k ranking across full and partial blocks.

---

## Usage

V1 and V2 share the same 3-function API. Replace `flash_topk_attn` with `flash_topk_attn_v2` to use the faster backend.

### Scoring

```python
import torch
from flash_topk_attn import flash_topk_score       # V1
# from flash_topk_attn_v2 import flash_topk_score  # V2 (same API, faster)

B, N, H, D = 2, 1024, 8, 64
q = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)

topk_indices, topk_scores = flash_topk_score(
    q, k,
    num_heads=H,
    score_block_size=64,
    topk=16,
)
# topk_indices: [B, H, N, topk] int32
# topk_scores:  [B, H, N, topk] float32
```

V1 additionally supports `score_only=False` to return attention output alongside scores:

```python
v = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
o, topk_indices, topk_scores = flash_topk_score(
    q, k, v, num_heads=H, score_block_size=64, topk=16,
)
# o: [B, N, C]  attention output (V1 only)
```

### Sparse Attention

```python
from flash_topk_attn import flash_topk_score, flash_topk_attn, build_qblock_topk_indices
# from flash_topk_attn_v2 import flash_topk_score, flash_topk_attn, build_qblock_topk_indices

# Step 1: Score — per-query top-k block indices
topk_indices, topk_scores = flash_topk_score(
    q, k, num_heads=H, score_block_size=64, topk=16,
)

# Step 2: Build — group per-query indices into per-q-block candidate sets
merged_indices, counts, S_MAX = build_qblock_topk_indices(
    topk_indices, q_block_size=32,
)
# merged_indices: [B, H, QM, S_MAX]  sorted unique block ids, -1 padded
# counts:         [B, H, QM]         valid count per q-block

# Step 3: Attend — sparse attention over candidate blocks
o_sparse, lse = flash_topk_attn(
    q, k, v, merged_indices, counts,
    num_heads=H, q_block_size=32, kv_block_size=64,
)
# o_sparse: [B, N, C]   sparse attention output
# lse:      [B, H, N]   log-sum-exp (float32)
```

`merged_indices` can be reused across multiple attention calls (e.g. different layers sharing the same sparsity pattern).

### Virtual Padding

```python
# N=1000 is not divisible by 64, pad to 1024
topk_indices, topk_scores = flash_topk_score(
    q, k, num_heads=H, score_block_size=64, topk=16,
    score_only=True, padding=(0, 24),  # score_only is V1 only
)

# Q-side and KV-side padding are independent
q_padding  = (8, 24)   # q_pad_head + N + q_pad_tail must divide q_block_size
kv_padding = (0, 24)   # kv_pad_head + N + kv_pad_tail must divide kv_block_size

merged_indices, counts, S_MAX = build_qblock_topk_indices(
    topk_indices, q_block_size=32, q_padding=q_padding,
)
o_sparse, lse = flash_topk_attn(
    q, k, v, merged_indices, counts,
    num_heads=H, q_block_size=32, kv_block_size=64,
    q_padding=q_padding, kv_padding=kv_padding,
)
```

---

## Performance

### V2 vs V1 End-to-End

V2 is **2x–4x faster** than V1 across all configurations (RTX 4090, bfloat16):

| B | N | H | D | Block Size | Top-K | V1 | V2 | Speedup |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 4,096 | 32 | 128 | 64 | 16 | 22.01 ms | 7.26 ms | **3.0x** |
| 1 | 8,192 | 32 | 128 | 64 | 16 | 87.25 ms | 28.15 ms | **3.1x** |
| 1 | 4,096 | 32 | 128 | 64 | 32 | 29.82 ms | 7.37 ms | **4.0x** |
| 2 | 4,096 | 32 | 128 | 64 | 16 | 43.44 ms | 14.42 ms | **3.0x** |
| 1 | 4,032 | 32 | 128 | 48 | 16 | 37.33 ms | 11.25 ms | **3.3x** |
| 1 | 5,056 | 32 | 128 | 64 | 16 | 33.97 ms | 11.42 ms | **3.0x** |
| 1 | 4,096 | 32 | 64 | 64 | 16 | 9.37 ms | 4.47 ms | **2.1x** |

Per-stage breakdown: [V2_BENCHMARK.md](V2_BENCHMARK.md)

### V1 Scoring vs Baselines

**Naive** = standard attention + `torch.topk` (two separate passes); **FA2** = Flash Attention 2 forward only, no TopK scoring (lower bound).

| Sequence Length | Heads | Block Size | Top-K | Speedup vs Naive | Overhead vs FA2 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2,048 | 8 | 64 | 16 | **4.5x faster** | 1.80x slower |
| 4,096 | 8 | 64 | 16 | **5.2x faster** | 1.96x slower |
| 8,192 | 4 | 128 | 16 | **5.3x faster** | 1.52x slower |
| 16,384 | 4 | 256 | 16 | — | 1.58x slower |
| 65,536 | 4 | 1,024 | 16 | — | 1.52x slower |
| 262,144 | 4 | 4,096 | 16 | — | 1.57x slower |

Naive materializes the full $N \times N$ attention matrix, causing GPU out-of-memory (OOM) at long sequences. Full benchmark results: [BENCHMARK.md](BENCHMARK.md)

---

## V2 Internals

V2 decomposes the fused V1 kernel into four operators with Triton + CUDA backend auto-dispatch:

1. **Block Scoring** — computes per-query per-block softmax probability mass $s_j$ (same as V1), but as a standalone operator without top-k selection
2. **Top-K Selection** — replaces V1's online Bitonic Sort with a two-step process: sort blocks by average score ($\text{avg} = \text{raw} / \text{valid\_count}$, ensuring fair ranking with virtual padding), then walk the sorted order accumulating raw scores until the cumulative sum reaches `threshold` or `max_topk` blocks are selected — whichever comes first
3. **Q-Block Merging** — replaces V1's exact set union with a score-weighted merge: scatter-add each query's top-k average scores into a shared $[M]$-sized accumulator per qblock, then sort descending and keep the top `qblock_topk` blocks
4. **Sparse Attention** — CUDA CuTe MMA flash attention over the merged block indices

Setting `threshold=1.0` and `qblock_topk = q_block_size * K` reproduces V1's exact behavior.

| Operator | Triton | CUDA | Auto rule |
|----------|:------:|:----:|-----------|
| Block Scoring | all D | D_kernel >= 80 | D_kernel < 80 → Triton |
| Top-K Selection | M_PAD <= 512 | M <= 4096 | M_PAD <= 128 → Triton |
| Q-Block Merging | — | always | CUDA only |
| Sparse Attention | — | always | CUDA only |

D_kernel is the head dimension padded to the nearest supported value in (32, 64, 96, 128, 160, 256).

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.5.0, < 2.7.0
- Triton >= 3.0.0
- CUDA-capable GPU

## Install

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## Todo

### V1
- [x] Flash Scoring kernel (fused scoring + top-k + attention)
- [x] Sparse Attention kernel (q-block shared candidates)

### V2
- [x] Block Scoring (Triton + CUDA auto-dispatch)
- [x] Top-K Selection (Triton + CUDA auto-dispatch)
- [x] Q-Block Merging (CUDA)
- [x] Sparse Attention (CUDA)
- [ ] Backward pass

## License

[MIT](LICENSE)

## Citation

```bibtex
@misc{flash-topk-attention,
  author = {YouHuang67},
  title  = {Flash TopK Attention},
  year   = {2025},
  url    = {https://github.com/YouHuang67/flash-topk-attention},
}
```
