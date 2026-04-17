"""JIT wrapper for sparse attention backward dK, dV kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"
_CUTLASS_INCLUDE = (
    "/home/hy/OpenRLHF/projects/flash-attention/csrc/cutlass/include"
)


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "sparse_attn_bwd_dkv_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "sparse_attn_bwd_dkv.cpp").read_text()
    return load_inline(
        name="sparse_attn_bwd_dkv_cuda",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_include_paths=[_CUTLASS_INCLUDE, str(_CSRC_DIR)],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-gencode", "arch=compute_80,code=sm_80",
            "-gencode", "arch=compute_86,code=sm_86",
        ],
        verbose=False,
    )


def flash_sparse_attn_bwd_dkv_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do_tensor: torch.Tensor,
    reverse_indices: torch.Tensor,
    reverse_counts: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_lse: torch.Tensor,
    delta: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    softmax_scale: float,
    q_pad_head: int,
    kv_pad_head: int,
    N_real: Optional[int] = None,
    sorted_kv_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch sparse attention backward dK, dV CUDA kernel.

    Args:
        q: ``[B, N_phys, H, D]`` bf16/fp16, contiguous.
        k: ``[B, N_phys, H, D]``.
        v: ``[B, N_phys, H, D]``.
        do_tensor: ``[B, N_phys, H, D]``.
        reverse_indices: ``[B, H, M, QM]`` int32.
        reverse_counts: ``[B, H, M]`` int32.
        softmax_max: ``[B, H, N_real]`` fp32.
        softmax_lse: ``[B, H, N_real]`` fp32.
        delta: ``[B, H, N_real]`` fp32.
        num_heads: Number of attention heads.
        q_block_size: Query block size.
        kv_block_size: KV block size.
        softmax_scale: Softmax scale.
        q_pad_head: Virtual Q head padding.
        kv_pad_head: Virtual KV head padding.
        N_real: Real sequence length.
        sorted_kv_indices: ``[B, H, M]`` int32, pre-computed sorted
            KV block indices by descending reverse_count. When None,
            computed internally via torch.argsort.

    Returns:
        (dk, dv): each ``[B, N_real, C]`` same dtype as q.
    """
    B, N_phys = q.shape[0], q.shape[1]
    H = num_heads
    D = q.shape[-1]
    N = N_real if N_real is not None else N_phys
    M = reverse_indices.shape[2]
    QM = reverse_indices.shape[3]
    C = H * D

    if sorted_kv_indices is None:
        sorted_kv_indices = torch.argsort(
            reverse_counts, dim=-1, descending=True
        ).to(torch.int32).contiguous()

    dk = torch.empty(B, N, H, D, dtype=q.dtype, device=q.device)
    dv = torch.empty(B, N, H, D, dtype=q.dtype, device=q.device)

    _get_ext().sparse_attn_bwd_dkv_launch(
        q, k, v, do_tensor,
        reverse_indices, reverse_counts,
        sorted_kv_indices,
        softmax_max, softmax_lse, delta,
        dk, dv,
        B, N, N_phys, H, D,
        QM, M,
        q_block_size, kv_block_size,
        softmax_scale,
        q_pad_head, kv_pad_head,
    )

    return dk.view(B, N, C), dv.view(B, N, C)


__all__ = ["flash_sparse_attn_bwd_dkv_cuda"]
