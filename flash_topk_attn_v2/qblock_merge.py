"""Qblock merge: aggregate per-query topk into per-qblock topk."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "qblock_merge_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "qblock_merge.cpp").read_text()
    return load_inline(
        name="qblock_merge_cuda",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        verbose=False,
    )


def _flash_qblock_merge_naive(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    q_block_size: int,
    qblock_topk: int,
    num_score_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation for qblock merge.

    Fully vectorized: scatter_add_ for aggregation, torch.topk for selection.
    No Python loops over batch/head/qblock dimensions.
    """
    if topk_indices.ndim != 4:
        raise ValueError("topk_indices must be [B, H, N, M_out]")
    if topk_scores.ndim != 4:
        raise ValueError("topk_scores must be [B, H, N, M_out]")

    B, H, N, M_out = topk_indices.shape
    if N % q_block_size != 0:
        raise ValueError(
            f"N ({N}) must be divisible by q_block_size ({q_block_size})"
        )

    QM = N // q_block_size
    flat = q_block_size * M_out

    idx = topk_indices.view(B, H, QM, flat).long()
    scr = topk_scores.float().view(B, H, QM, flat)

    valid = idx >= 0
    scr_masked = torch.where(valid, scr, torch.zeros_like(scr))
    idx_safe = torch.where(valid, idx, torch.zeros_like(idx))

    agg = torch.zeros(
        B, H, QM, num_score_blocks,
        dtype=torch.float32, device=scr.device,
    )
    agg.scatter_add_(3, idx_safe, scr_masked)

    actual_k = min(qblock_topk, num_score_blocks)
    topk_vals, topk_order = agg.topk(actual_k, dim=-1)

    invalid = topk_vals <= 0.0
    merged_scores = topk_vals.masked_fill(invalid, 0.0)
    merged_indices = topk_order.masked_fill(invalid, -1).to(torch.int32)

    if qblock_topk > num_score_blocks:
        pad_k = qblock_topk - num_score_blocks
        merged_indices = torch.nn.functional.pad(
            merged_indices, (0, pad_k), value=-1,
        )
        merged_scores = torch.nn.functional.pad(
            merged_scores, (0, pad_k), value=0.0,
        )

    return merged_indices, merged_scores


def flash_qblock_merge(
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    q_block_size: int,
    qblock_topk: int,
    num_score_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Merge per-query topk into per-qblock topk via scatter-add + sort.

    Args:
        topk_indices: ``[B, H, N, M_out]`` int32, ``-1`` for padding.
        topk_scores: ``[B, H, N, M_out]`` float32 (avg scores).
        q_block_size: Number of queries per qblock.
        qblock_topk: Number of top blocks to keep per qblock.
        num_score_blocks: Total number of kv score blocks (M_total).

    Returns:
        A tuple of two tensors:
            - **merged_indices** ``[B, H, QM, qblock_topk]`` int32, ``-1`` padding.
            - **merged_scores** ``[B, H, QM, qblock_topk]`` float32, ``0`` padding.
    """
    if topk_indices.ndim != 4:
        raise ValueError("topk_indices must be [B, H, N, M_out]")
    if topk_scores.ndim != 4:
        raise ValueError("topk_scores must be [B, H, N, M_out]")
    if topk_indices.device.type != "cuda":
        raise ValueError("flash_qblock_merge requires CUDA tensors")

    B, H, N, M_out = topk_indices.shape
    if N % q_block_size != 0:
        raise ValueError(
            f"N ({N}) must be divisible by q_block_size ({q_block_size})"
        )
    if num_score_blocks <= 0:
        raise ValueError("num_score_blocks must be positive")
    if num_score_blocks > 4096:
        raise NotImplementedError(
            f"num_score_blocks={num_score_blocks} exceeds CUDA block sort "
            f"limit 4096"
        )
    if qblock_topk <= 0:
        raise ValueError("qblock_topk must be positive")

    QM = N // q_block_size
    num_slices = B * H * QM

    indices_in = topk_indices.reshape(B * H, QM, q_block_size, M_out)
    indices_in = indices_in.reshape(num_slices, q_block_size * M_out).contiguous()
    scores_in = topk_scores.float().reshape(B * H, QM, q_block_size, M_out)
    scores_in = scores_in.reshape(num_slices, q_block_size * M_out).contiguous()

    merged_indices = torch.full(
        (num_slices, qblock_topk), -1,
        dtype=torch.int32, device=topk_indices.device,
    )
    merged_scores = torch.zeros(
        num_slices, qblock_topk,
        dtype=torch.float32, device=topk_indices.device,
    )

    ext = _get_ext()
    ext.qblock_merge_launch(
        indices_in, scores_in,
        merged_indices, merged_scores,
        q_block_size, M_out, num_score_blocks,
        qblock_topk, num_slices,
    )

    merged_indices = merged_indices.view(B, H, QM, qblock_topk)
    merged_scores = merged_scores.view(B, H, QM, qblock_topk)
    return merged_indices, merged_scores


__all__ = ["flash_qblock_merge", "_flash_qblock_merge_naive"]
