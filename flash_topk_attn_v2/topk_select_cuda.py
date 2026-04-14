"""CUB-based top-k block selection for M <= 4096."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "topk_select_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "topk_select.cpp").read_text()
    return load_inline(
        name="topk_select_cuda",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        verbose=False,
    )


def flash_topk_select_cuda(
    block_scores: torch.Tensor,
    threshold: float,
    max_topk: int,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort block scores by avg, cumsum raw scores, and truncate at threshold.

    Uses CUB BlockRadixSort for in-block descending sort (M <= 4096).
    Drop-in replacement for ``flash_topk_select`` when M > 512.

    Args:
        block_scores: Per-query block scores, shape ``[B, H, N, M]``.
            Accepts float32/float16/bfloat16 (internally cast to float32).
        threshold: Cumulative raw-score threshold. Blocks are kept until
            the running sum of sorted raw scores reaches this value.
        max_topk: Maximum number of blocks to keep per query. 0 means
            no limit (keep up to M).
        score_block_size: Number of tokens per score block, used to
            convert raw scores to avg scores (``avg = raw / valid_count``).
        padding: ``(pad_head, pad_tail)`` — number of invalid tokens at
            the first and last block. Reduces their valid_count accordingly.

    Returns:
        A tuple of three tensors, each ``[B, H, N, M_OUT]``:
            - **topk_indices** (int32): Original block indices in descending
              avg-score order. ``-1`` marks positions beyond the cutoff.
            - **topk_raw_scores** (float32): Corresponding raw scores.
              ``0.0`` beyond the cutoff.
            - **topk_avg_scores** (float32): Corresponding avg scores.
              ``0.0`` beyond the cutoff.

        where ``M_OUT = min(max_topk, M)`` if ``max_topk > 0``, else ``M``.
    """
    if block_scores.ndim != 4:
        raise ValueError("block_scores must be [B, H, N, M]")
    if block_scores.device.type != "cuda":
        raise ValueError("flash_topk_select_cuda requires CUDA tensor")
    if block_scores.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("block_scores must be float16/bfloat16/float32")

    B, H, N, M = block_scores.shape
    if M <= 0:
        raise ValueError("M must be positive")
    if M > 4096:
        raise NotImplementedError(f"M={M} exceeds CUDA block sort limit 4096")
    if score_block_size <= 0:
        raise ValueError("score_block_size must be positive")

    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= score_block_size or pad_tail >= score_block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < "
            f"score_block_size ({score_block_size})"
        )
    if max_topk < 0:
        raise ValueError("max_topk must be >= 0")

    m_out = min(max_topk, M) if max_topk > 0 else M
    num_slices = B * H * N

    if block_scores.dtype == torch.float32:
        block_in = block_scores.contiguous()
    else:
        block_in = block_scores.float().contiguous()
    block_in = block_in.view(num_slices, M)

    topk_indices = torch.empty(
        num_slices, m_out, device=block_scores.device, dtype=torch.int32)
    topk_raw = torch.empty(
        num_slices, m_out, device=block_scores.device, dtype=torch.float32)
    topk_avg = torch.empty(
        num_slices, m_out, device=block_scores.device, dtype=torch.float32)

    ext = _get_ext()
    ext.topk_select_launch(
        block_in, topk_indices, topk_raw, topk_avg,
        M, m_out, score_block_size, pad_head, pad_tail,
        float(threshold), num_slices,
    )

    topk_indices = topk_indices.view(B, H, N, m_out)
    topk_raw = topk_raw.view(B, H, N, m_out)
    topk_avg = topk_avg.view(B, H, N, m_out)
    return topk_indices, topk_raw, topk_avg


__all__ = ["flash_topk_select_cuda"]
