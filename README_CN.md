[English](README.md) | 中文

# Flash TopK Attention

融合 Flash Attention、top-k KV block 打分与稀疏注意力的内核库。

- **V1** (`flash_topk_attn`)：单个融合 Triton 内核，在一次 KV 遍历中完成 block 打分、top-k 选取和注意力输出。
- **V2** (`flash_topk_attn_v2`)：与 V1 相同的 API，**2x–4x 加速**。内部使用模块化的 Triton + CUDA 流水线。支持反向传播（dQ, dK, dV），由专用 CUDA 内核实现。

---

## 算法

### 打分（Scoring）

标准 Flash Attention 利用 online softmax 技巧计算 $O = \text{softmax}(QK^\top / \sqrt{D})\ V$，通过逐 block 迭代 KV 避免展开完整的 $N \times N$ 矩阵。其中 $B$ 为 batch size，$N$ 为序列长度，$H$ 为注意力头数，$D$ 为每头维度。

本内核在此基础上将 KV 序列划分为 $M = N / b$ 个不重叠的 block，每个 block 大小为 $b$，并为每个 block $j$ 计算**块级注意力分数**：

$$s_j = \sum_{i \ \in\  \text{block}_j} p_i, \qquad p_i = \frac{\exp(q_t k_i^\top / \sqrt{D})}{\sum_{l=1}^{N} \exp(q_t k_l^\top / \sqrt{D})}.$$

随后选取分数最高的 $\text{top-}k$ 个 block 索引：

$$\mathbf{I} = \underset{j \in \{1,\ldots,M\}}{\text{argtop-}k}\ s_j,$$

输出 $\mathbf{I}$ 的形状为 $[B, H, N, k]$，可驱动后续稀疏注意力计算，将每 token 的 KV 访问量从 $O(N)$ 降至 $O(k \cdot b)$。

### 稀疏注意力（Sparse Attention）

基于打分阶段得到的 $\text{top-}k$ block 索引 $\mathbf{I}$，稀疏注意力仅对选中的 KV block 计算输出。

**Q-Block 共享候选机制**：将 query 按大小 $g$（`q_block_size`）分组。对于第 $m$ 个 query 块（覆盖 query $[mg,\ (m+1)g)$），取组内所有 query 的 top-k 索引的并集，构造共享候选集：

$$\mathcal{C}_m = \bigcup_{t=mg}^{(m+1)g-1} \mathrm{TopK}(q_t)$$

候选按 block id 升序排列，记 $L_m = |\mathcal{C}_m| \leq g \cdot k$。

组内每个 query $q_t$ 对**整个**共享候选集 $\mathcal{C}_m$ 计算注意力：

$$O_t = \sum_{j \in \mathcal{C}_m} \frac{\exp(q_t k_j^\top / \sqrt{D})}{\sum_{j' \in \mathcal{C}_m} \exp(q_t k_{j'}^\top / \sqrt{D})} \cdot v_j$$

- $g=1$：每个 query 仅访问自己的 top-k blocks（per-query 稀疏注意力）
- $g>1$：query 共享候选，实现批量 KV 访问，但每个 query 的注意力范围略有扩大

### 虚拟 Padding

当 N 不能整除 block 大小 b 时，虚拟 padding 将序列扩展为 N' = pad_head + N + pad_tail，使 N' 能被 b 整除。Padding 纯为逻辑概念，QKV 数据不变，padding 位置在 softmax 前被掩码为负无穷。对于头尾的不完整 block，block 分数按有效 token 数归一化，保证完整 block 与部分 block 之间 top-k 排序公平。

---

## 使用方法

### V1

三个函数：`flash_topk_score` 用于 block 级打分，`build_qblock_topk_indices` 用于 q-block 合并，`flash_topk_attn` 用于稀疏注意力。

```python
import torch
from flash_topk_attn import flash_topk_score, flash_topk_attn, build_qblock_topk_indices

B, N, H, D = 2, 1024, 8, 64
q = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
k = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)
v = torch.randn(B, N, H * D, device="cuda", dtype=torch.float16)

# 第 1 步：打分，获取 per-query top-k block 索引
topk_indices, topk_scores = flash_topk_score(
    q, k, num_heads=H, score_block_size=64, topk=16,
)
# topk_indices: [B, H, N, topk] int32
# topk_scores:  [B, H, N, topk] float32

# 第 2 步：构建，按 q-block 聚合 per-query 索引为共享候选集
merged_indices, counts, S_MAX = build_qblock_topk_indices(
    topk_indices, q_block_size=32,
)
# merged_indices: [B, H, QM, S_MAX]  排序去重的 block id，-1 填充
# counts:         [B, H, QM]         每个 q-block 的有效索引数

# 第 3 步：注意力，仅对候选 block 计算稀疏注意力
o_sparse, lse = flash_topk_attn(
    q, k, v, merged_indices, counts,
    num_heads=H, q_block_size=32, kv_block_size=64,
)
# o_sparse: [B, N, C]   稀疏注意力输出
# lse:      [B, H, N]   log-sum-exp (float32)
```

`merged_indices` 可跨多次注意力调用复用（如多层共享相同稀疏模式）。

V1 还支持 `score_only=False`，在打分的同时返回注意力输出：

```python
o, topk_indices, topk_scores = flash_topk_score(
    q, k, v, num_heads=H, score_block_size=64, topk=16,
)
# o: [B, N, C]  注意力输出（仅 V1）
```

### V2

V2 提供相同的三个函数。将 `flash_topk_attn` 替换为 `flash_topk_attn_v2` 即可使用更快的 Triton + CUDA 后端。默认参数下 V2 输出与 V1 完全一致。

V2 在打分和合并阶段各新增一个控制参数。V1 按原始分数 $s_j$ 排序取 top $k$ 个 block。V2 先按均值分数 $s_j / n_j$（$n_j$ 为 block $j$ 的有效 token 数）排序，然后沿排序顺序累积原始分数，直到 $\sum s_j \geq$ `threshold` 或达到 $k$ 个 block。`threshold=1.0`（默认）保留所有 top block，与 V1 一致。合并阶段，V1 将 q-block 内所有 query 的 top-k 索引取并集构造候选集 $\mathcal{C}_m$。V2 改为将各 query 的均值分数 $s_j / n_j$ scatter-add 到共享累加器，降序排序后取前 `qblock_topk` 个 block。`qblock_topk` $= g \cdot k$（默认）保留所有唯一 block，与 V1 一致。

```python
from flash_topk_attn_v2 import flash_topk_score, flash_topk_attn, build_qblock_topk_indices

# 第 1 步：打分，带 threshold 截断
topk_indices, topk_raw_scores, topk_avg_scores = flash_topk_score(
    q, k, num_heads=H, score_block_size=64, topk=16,
    threshold=0.9,
)
# topk_indices:    [B, H, N, topk] int32
# topk_raw_scores: [B, H, N, topk] float32
# topk_avg_scores: [B, H, N, topk] float32

# 第 2 步：构建，score-weighted 合并 + qblock_topk 截断
merged_indices, counts, S_MAX = build_qblock_topk_indices(
    topk_indices, q_block_size=32,
    topk_scores=topk_avg_scores, qblock_topk=128,
)

# 第 3 步：注意力（与 V1 相同）
o_sparse, lse = flash_topk_attn(
    q, k, v, merged_indices, counts,
    num_heads=H, q_block_size=32, kv_block_size=64,
)
```

V2 稀疏注意力支持通过 `torch.autograd` 反向传播。梯度 dQ、dK、dV 由专用 CUTLASS/CuTe CUDA 内核（SM80+）计算，通过反向索引构建（KV-block → Q-blocks）实现高效 dK/dV 累积。

### 虚拟 Padding

当 N 不能整除 block 大小时，使用 `padding` 指定虚拟 padding。

```python
# N=1000 不能整除 64，pad 到 1024
topk_indices, topk_scores = flash_topk_score(
    q, k, num_heads=H, score_block_size=64, topk=16,
    score_only=True, padding=(0, 24),  # score_only 仅 V1
)

# Q 侧与 KV 侧 padding 独立指定
q_padding  = (8, 24)   # q_pad_head + N + q_pad_tail 须整除 q_block_size
kv_padding = (0, 24)   # kv_pad_head + N + kv_pad_tail 须整除 kv_block_size

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

## 性能

V2 将 V1 的融合内核拆分为四个算子（block scoring, top-k selection, q-block merging, sparse attention），各自自动选择 Triton / CUDA 后端。

| 算子 | Triton | CUDA | 自动规则 |
|------|:------:|:----:|---------|
| Block Scoring | 所有 D | D_kernel >= 80 | D_kernel < 80 用 Triton |
| Top-K Selection | M_PAD <= 512 | M <= 4096 | M_PAD <= 128 用 Triton |
| Q-Block Merging | 无 | 始终 | 仅 CUDA |
| Sparse Attention | 无 | 始终 | 仅 CUDA |

D_kernel 为 head 维度向上取整到最近的支持值 (32, 64, 96, 128, 160, 256)。

### V2 vs V1 端到端

V2 在所有配置下比 V1 快 **2x–4x**（RTX 4090, bfloat16）：

| B | N | H | D | Block Size | Top-K | V1 | V2 | 加速比 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | 4,096 | 32 | 128 | 64 | 16 | 22.01 ms | 7.26 ms | **3.0x** |
| 1 | 8,192 | 32 | 128 | 64 | 16 | 87.25 ms | 28.15 ms | **3.1x** |
| 1 | 4,096 | 32 | 128 | 64 | 32 | 29.82 ms | 7.37 ms | **4.0x** |
| 2 | 4,096 | 32 | 128 | 64 | 16 | 43.44 ms | 14.42 ms | **3.0x** |
| 1 | 4,032 | 32 | 128 | 48 | 16 | 37.33 ms | 11.25 ms | **3.3x** |
| 1 | 5,056 | 32 | 128 | 64 | 16 | 33.97 ms | 11.42 ms | **3.0x** |
| 1 | 4,096 | 32 | 64 | 64 | 16 | 9.37 ms | 4.47 ms | **2.1x** |

各阶段详细数据：[V2_BENCHMARK_CN.md](V2_BENCHMARK_CN.md)

### V2 稀疏注意力反向传播

CUDA 反向（CUTLASS/CuTe dQ + dKV 内核）vs naive 反向（PyTorch autograd 对密集 masked attention 自动求导）。RTX 3090, bfloat16, `q_block_size=kv_block_size=64`：

| 配置 | CUDA bwd (ms) | Naive total (ms) | 加速比 |
|:-:|:-:|:-:|:-:|
| N=256 H=8 D=128 topk=8 | 2.09 | 13.15 | **6x** |
| N=2048 H=8 D=128 topk=8 | 1.73 | 149.77 | **79x** |
| N=8192 H=2 D=128 topk=8 | 1.47 | 194.21 | **119x** |
| N=2048 H=8 D=128 topk=32 | 1.56 | 549.14 | **275x** |
| N=2048 H=8 D=256 topk=8 | 1.57 | 148.47 | **76x** |
| B=4 H=8 N=2048 D=128 topk=8 | 1.83 | 575.89 | **249x** |

梯度精度（与 naive 对比，median 绝对误差）：dQ < 2.4e-4, dK < 9.1e-5, dV < 1.3e-3。完整数据：[V2_BENCHMARK_CN.md](V2_BENCHMARK_CN.md)

### V1 打分 vs 基线

**Naive** = 标准注意力 + `torch.topk`（两次独立 pass）；**FA2** = 仅 Flash Attention 2 前向，不含 TopK 计分（性能下界）。

| 序列长度 | 注意力头数 | Block 大小 | Top-K | 对比 Naive 加速比 | 对比 FA2 额外开销 |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 2,048 | 8 | 64 | 16 | **4.5x 加速** | 1.80x 减速 |
| 4,096 | 8 | 64 | 16 | **5.2x 加速** | 1.96x 减速 |
| 8,192 | 4 | 128 | 16 | **5.3x 加速** | 1.52x 减速 |
| 16,384 | 4 | 256 | 16 | — | 1.58x 减速 |
| 65,536 | 4 | 1,024 | 16 | — | 1.52x 减速 |
| 262,144 | 4 | 4,096 | 16 | — | 1.57x 减速 |

Naive 需要展开完整的 $N \times N$ 注意力矩阵，在长序列下会导致 GPU 显存不足 (OOM)。完整 benchmark：[BENCHMARK.md](BENCHMARK.md)

---

## 环境要求

- Python >= 3.10
- PyTorch >= 2.5.0, < 2.7.0
- Triton >= 3.0.0
- 支持 CUDA 的 GPU

## 安装

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

## 许可证

[MIT](LICENSE)

## 引用

```bibtex
@misc{flash-topk-attention,
  author = {YouHuang67},
  title  = {Flash TopK Attention},
  year   = {2025},
  url    = {https://github.com/YouHuang67/flash-topk-attention},
}
```
