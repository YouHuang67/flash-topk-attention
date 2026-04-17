"""JIT wrapper for build_reverse_indices CUDA kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "build_reverse_indices_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "build_reverse_indices.cpp").read_text()
    return load_inline(
        name="build_reverse_indices_cuda",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
        ],
        verbose=False,
    )


def flash_build_reverse_indices_cuda(
    merged_indices: torch.Tensor,
    counts: torch.Tensor,
    num_kv_blocks: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build reverse indices and sorted KV indices (CUDA).

    Fuses Python-side build_reverse_indices + torch.argsort into
    two CUDA kernels: Phase 1 atomic scatter + Phase 2 CUB
    BlockRadixSort per-(b,h).

    Args:
        merged_indices: ``[B, H, QM, S_MAX]`` int32, -1 padding.
        counts: ``[B, H, QM]`` int32.
        num_kv_blocks: Total number of KV blocks (M).

    Returns:
        (reverse_indices, reverse_counts, sorted_kv_indices):
            - reverse_indices: ``[B, H, M, QM]`` int32, -1 padding.
            - reverse_counts: ``[B, H, M]`` int32.
            - sorted_kv_indices: ``[B, H, M]`` int32, descending by
              reverse_count.
    """
    B, H, QM, S_MAX = merged_indices.shape
    M = num_kv_blocks
    device = merged_indices.device

    reverse_indices = torch.full(
        (B, H, M, QM), -1, dtype=torch.int32, device=device,
    )
    reverse_counts = torch.zeros(
        B, H, M, dtype=torch.int32, device=device,
    )
    sorted_kv_indices = torch.empty(
        B, H, M, dtype=torch.int32, device=device,
    )

    _get_ext().build_reverse_indices_launch(
        merged_indices.contiguous(),
        counts.contiguous(),
        reverse_indices,
        reverse_counts,
        sorted_kv_indices,
        B, H, QM, S_MAX, M,
    )

    return reverse_indices, reverse_counts, sorted_kv_indices


__all__ = ["flash_build_reverse_indices_cuda"]
