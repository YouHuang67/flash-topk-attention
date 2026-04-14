"""V1-compatible public API for flash_topk_attn_v2."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from flash_topk_attn_v2.block_score import flash_block_score
from flash_topk_attn_v2.topk_select import flash_topk_select
from flash_topk_attn_v2.qblock_merge import flash_qblock_merge
from flash_topk_attn_v2.sparse_attn import flash_sparse_attn


def flash_topk_score(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: Optional[int] = None,
    score_block_size: int = 64,
    topk: int = 16,
    padding: Tuple[int, int] = (0, 0),
    threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-level top-k scoring with Triton + CUDA auto-dispatch.

    Args:
        q: Query ``[B, N, C]`` or ``[B, H, N, D]`` on CUDA.
        k: Key, same layout and dtype as ``q``.
        num_heads: Required for 3D input; optional for 4D.
        score_block_size: Tokens per score block.
        topk: Maximum number of top blocks to select per query.
        padding: ``(pad_head, pad_tail)`` virtual padding.
        threshold: Cumulative raw-score cutoff (V2 only). Blocks are
            kept until cumulative raw score reaches this value or
            ``topk`` blocks are selected. Defaults to 1.0 (keep all
            top blocks up to ``topk``, equivalent to V1).

    Returns:
        (topk_indices, topk_raw_scores, topk_avg_scores):
            - topk_indices: ``[B, H, N, topk]`` int32.
            - topk_raw_scores: ``[B, H, N, topk]`` float32.
            - topk_avg_scores: ``[B, H, N, topk]`` float32.
    """
    input_4d = q.ndim == 4
    if input_4d:
        B, H, N, D = q.shape
        if num_heads is not None and num_heads != H:
            raise ValueError(f"num_heads ({num_heads}) != q.shape[1] ({H})")
        num_heads = H
    else:
        if q.ndim != 3:
            raise ValueError("q must be 3D [B, N, C] or 4D [B, H, N, D]")
        if num_heads is None:
            raise ValueError("num_heads is required for 3D [B, N, C] input")

    block_scores = flash_block_score(
        q, k, num_heads=num_heads,
        score_block_size=score_block_size, padding=padding,
    )
    topk_indices, topk_raw, topk_avg = flash_topk_select(
        block_scores, threshold=threshold, max_topk=topk,
        score_block_size=score_block_size, padding=padding,
    )
    return topk_indices, topk_raw, topk_avg


def build_qblock_topk_indices(
    topk_indices: torch.Tensor,
    q_block_size: int,
    q_padding: Tuple[int, int] = (0, 0),
    topk_scores: Optional[torch.Tensor] = None,
    qblock_topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build per-qblock merged indices from per-query topk indices.

    Args:
        topk_indices: ``[B, H, N, K]`` int32, ``-1`` for padding.
        q_block_size: Queries per qblock.
        q_padding: ``(q_pad_head, q_pad_tail)`` virtual Q padding.
        topk_scores: ``[B, H, N, K]`` float32 (avg scores from
            ``flash_topk_score``). When provided, q-block merging uses
            score-weighted scatter-add. When ``None``, falls back to
            uniform weights (V1 equivalent).
        qblock_topk: Number of top blocks to keep per q-block after
            merging. Defaults to ``q_block_size * K`` (V1 equivalent,
            keeps all unique blocks).

    Returns:
        (merged_indices, counts, S_MAX):
            - merged_indices: ``[B, H, QM, S_MAX]`` int32, ``-1`` padding.
            - counts: ``[B, H, QM]`` int32.
            - S_MAX: int, last dim of merged_indices.
    """
    if topk_indices.ndim != 4:
        raise ValueError("topk_indices must be [B, H, N, K]")

    B, H, N, K = topk_indices.shape
    q_pad_head, q_pad_tail = q_padding
    device = topk_indices.device

    if q_pad_head < 0 or q_pad_tail < 0:
        raise ValueError("q_padding must be non-negative")
    if q_pad_head >= q_block_size:
        raise ValueError(
            f"q_pad_head={q_pad_head} must be < q_block_size={q_block_size}"
        )
    if q_pad_tail >= q_block_size:
        raise ValueError(
            f"q_pad_tail={q_pad_tail} must be < q_block_size={q_block_size}"
        )

    n_padded = q_pad_head + N + q_pad_tail
    if n_padded % q_block_size != 0:
        raise ValueError(
            f"q_pad_head + N + q_pad_tail = {n_padded} must be divisible "
            f"by q_block_size ({q_block_size})"
        )

    idx = topk_indices.to(torch.int32).contiguous()
    if q_pad_head > 0 or q_pad_tail > 0:
        idx = F.pad(idx, (0, 0, q_pad_head, q_pad_tail), value=-1)

    if qblock_topk is None:
        qblock_topk = q_block_size * K

    valid_mask = idx >= 0
    if valid_mask.any():
        num_score_blocks = idx[valid_mask].max().item() + 1
    else:
        num_score_blocks = 1

    if topk_scores is not None:
        scores = topk_scores.float().contiguous()
        if q_pad_head > 0 or q_pad_tail > 0:
            scores = F.pad(scores, (0, 0, q_pad_head, q_pad_tail), value=0.0)
    else:
        scores = torch.ones(idx.shape, dtype=torch.float32, device=device)
        scores[~valid_mask] = 0.0

    merged_indices, _merged_scores = flash_qblock_merge(
        idx, scores,
        q_block_size=q_block_size,
        qblock_topk=qblock_topk,
        num_score_blocks=num_score_blocks,
    )

    counts = (merged_indices >= 0).sum(dim=-1).int()
    S_MAX = merged_indices.shape[-1]
    return merged_indices, counts, S_MAX


def flash_topk_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_indices: torch.Tensor,
    counts: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    scale: Optional[float] = None,
    num_kv_heads: Optional[int] = None,
    kv_padding: Tuple[int, int] = (0, 0),
    q_padding: Tuple[int, int] = (0, 0),
    S_MAX: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention over merged KV block indices.

    Args:
        q: ``[B, N, C]`` on CUDA.
        k: ``[B, N, C]``.
        v: ``[B, N, C]``.
        merged_indices: ``[B, H, QM, S_MAX]`` int32 from
            ``build_qblock_topk_indices``.
        counts: ``[B, H, QM]`` int32 (ignored; inferred internally).
        num_heads: Number of attention heads.
        q_block_size: Query block size.
        kv_block_size: KV block size (must match score_block_size).
        scale: Softmax scale. Defaults to ``1 / sqrt(D)``.
        num_kv_heads: Must equal num_heads (GQA not supported).
        kv_padding: ``(kv_pad_head, kv_pad_tail)`` virtual KV padding.
        q_padding: ``(q_pad_head, q_pad_tail)`` virtual Q padding.
        S_MAX: Ignored; inferred from merged_indices shape.

    Returns:
        (output, lse):
            - output: ``[B, N, C]``, same dtype as q.
            - lse: ``[B, H, N]`` float32.
    """
    if num_kv_heads is not None and num_kv_heads != num_heads:
        raise NotImplementedError(
            "GQA (num_kv_heads != num_heads) is not supported"
        )

    return flash_sparse_attn(
        q, k, v, merged_indices,
        num_heads=num_heads,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        scale=scale,
        q_padding=q_padding,
        kv_padding=kv_padding,
    )
