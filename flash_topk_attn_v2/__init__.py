"""Public API for flash_topk_attn_v2."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from flash_topk_attn_v2.block_score import flash_block_score
from flash_topk_attn_v2.qblock_merge import flash_qblock_merge
from flash_topk_attn_v2.sparse_attn import flash_sparse_attn
from flash_topk_attn_v2.topk_select import flash_topk_select


def flash_topk_score(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: Optional[int] = None,
    score_block_size: int = 64,
    topk: int = 16,
    padding: Tuple[int, int] = (0, 0),
    threshold: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-level top-k scoring with Triton + CUDA auto-dispatch."""
    input_4d = q.ndim == 4
    if input_4d:
        _, head_count, _, _ = q.shape
        if num_heads is not None and num_heads != head_count:
            raise ValueError(
                f"num_heads ({num_heads}) != q.shape[1] ({head_count})"
            )
        num_heads = head_count
    else:
        if q.ndim != 3:
            raise ValueError("q must be 3D [B, N, C] or 4D [B, H, N, D]")
        if num_heads is None:
            raise ValueError("num_heads is required for 3D [B, N, C] input")

    block_scores = flash_block_score(
        q,
        k,
        num_heads=num_heads,
        score_block_size=score_block_size,
        padding=padding,
    )
    topk_indices, topk_raw_scores, topk_avg_scores = flash_topk_select(
        block_scores,
        threshold=threshold,
        max_topk=topk,
        score_block_size=score_block_size,
        padding=padding,
    )
    return topk_indices, topk_raw_scores, topk_avg_scores


def build_qblock_topk_indices(
    topk_indices: torch.Tensor,
    q_block_size: int,
    q_padding: Tuple[int, int] = (0, 0),
    topk_scores: Optional[torch.Tensor] = None,
    qblock_topk: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Build per-qblock merged indices from per-query topk indices."""
    if topk_indices.ndim != 4:
        raise ValueError("topk_indices must be [B, H, N, K]")

    _, _, sequence_length, topk_width = topk_indices.shape
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

    padded_sequence_length = q_pad_head + sequence_length + q_pad_tail
    if padded_sequence_length % q_block_size != 0:
        raise ValueError(
            f"q_pad_head + N + q_pad_tail = {padded_sequence_length} must be divisible "
            f"by q_block_size ({q_block_size})"
        )

    indices_int = topk_indices.to(torch.int32).contiguous()
    if q_pad_head > 0 or q_pad_tail > 0:
        indices_int = F.pad(
            indices_int,
            (0, 0, q_pad_head, q_pad_tail),
            value=-1,
        )

    if qblock_topk is None:
        qblock_topk = q_block_size * topk_width

    valid_mask = indices_int >= 0
    if valid_mask.any():
        num_score_blocks = indices_int[valid_mask].max().item() + 1
    else:
        num_score_blocks = 1

    if topk_scores is not None:
        scores_float = topk_scores.float().contiguous()
        if q_pad_head > 0 or q_pad_tail > 0:
            scores_float = F.pad(
                scores_float,
                (0, 0, q_pad_head, q_pad_tail),
                value=0.0,
            )
    else:
        scores_float = torch.ones(
            indices_int.shape,
            dtype=torch.float32,
            device=device,
        )
        scores_float[~valid_mask] = 0.0

    merged_indices, _merged_scores = flash_qblock_merge(
        indices_int,
        scores_float,
        q_block_size=q_block_size,
        qblock_topk=qblock_topk,
        num_score_blocks=num_score_blocks,
    )

    counts = (merged_indices >= 0).sum(dim=-1).int()
    max_merged_width = merged_indices.shape[-1]
    return merged_indices, counts, max_merged_width


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
    """Sparse attention over merged KV block indices."""
    del counts, S_MAX
    if num_kv_heads is not None and num_kv_heads != num_heads:
        raise NotImplementedError(
            "GQA (num_kv_heads != num_heads) is not supported"
        )

    return flash_sparse_attn(
        q,
        k,
        v,
        merged_indices,
        num_heads=num_heads,
        q_block_size=q_block_size,
        kv_block_size=kv_block_size,
        scale=scale,
        q_padding=q_padding,
        kv_padding=kv_padding,
    )


__all__ = [
    "flash_topk_score",
    "build_qblock_topk_indices",
    "flash_topk_attn",
]
