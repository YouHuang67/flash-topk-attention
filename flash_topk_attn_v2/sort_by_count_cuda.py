"""JIT wrapper for CUB-based sort_by_count kernel."""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "sort_by_count_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "sort_by_count.cpp").read_text()
    return load_inline(
        name="sort_by_count_cuda",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
        ],
        verbose=False,
    )


def flash_sort_by_count_cuda(
    reverse_counts: torch.Tensor,
    num_heads: int,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sort KV blocks globally by reverse_count descending (CUDA).

    Args:
        reverse_counts: ``[B, H, M]`` int32.
        num_heads: Number of attention heads.
        num_kv_blocks: Number of KV blocks (M).

    Returns:
        (sorted_counts, sorted_global_ids):
            - sorted_counts: ``[B*H*M]`` int32, descending.
            - sorted_global_ids: ``[B*H*M]`` int32.
    """
    B = reverse_counts.shape[0]
    total = B * num_heads * num_kv_blocks

    counts_flat = reverse_counts.reshape(-1).contiguous()

    max_val = counts_flat.max().item()
    end_bit = max(1, math.ceil(math.log2(max_val + 1))) if max_val > 0 else 1

    sorted_counts = torch.empty_like(counts_flat)
    sorted_global_ids = torch.empty(
        total, dtype=torch.int32, device=reverse_counts.device
    )

    _get_ext().sort_by_count_launch(
        counts_flat, sorted_counts, sorted_global_ids, total, end_bit
    )

    return sorted_counts, sorted_global_ids


__all__ = ["flash_sort_by_count_cuda"]
