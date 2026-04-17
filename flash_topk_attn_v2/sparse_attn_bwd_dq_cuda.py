"""JIT compilation wrapper for sparse attention backward dQ kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"
_CUTLASS_INCLUDE = (
    "/home/hy/OpenRLHF/projects/flash-attention/csrc/cutlass/include"
)


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "sparse_attn_bwd_dq_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "sparse_attn_bwd_dq.cpp").read_text()
    return load_inline(
        name="sparse_attn_bwd_dq_cuda",
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


def flash_sparse_attn_bwd_dq_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do_tensor: torch.Tensor,
    merged_indices: torch.Tensor,
    counts: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_lse: torch.Tensor,
    delta: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    qblock_topk: int,
    softmax_scale: float,
    q_pad_head: int,
    kv_pad_head: int,
    N_real: Optional[int] = None,
) -> torch.Tensor:
    """Launch sparse attention backward dQ CUDA kernel.

    Args:
        q: ``[B, N_phys, H, D]`` bf16/fp16, contiguous.
        k: ``[B, N_phys, H, D]``.
        v: ``[B, N_phys, H, D]``.
        do_tensor: ``[B, N_phys, H, D]``.
        merged_indices: ``[B, H, QM, qblock_topk]`` int32.
        counts: ``[B, H, QM]`` int32.
        softmax_max: ``[B, H, N_real]`` fp32.
        softmax_lse: ``[B, H, N_real]`` fp32.
        delta: ``[B, H, N_real]`` fp32.
        num_heads: Number of attention heads.
        q_block_size: Query block size.
        kv_block_size: KV block size.
        qblock_topk: S_MAX dimension.
        softmax_scale: Softmax scale.
        q_pad_head: Virtual Q head padding.
        kv_pad_head: Virtual KV head padding.
        N_real: Real sequence length.

    Returns:
        dq: ``[B, N_real, C]`` same dtype as q.
    """
    B, N_phys = q.shape[0], q.shape[1]
    H = num_heads
    D = q.shape[-1]
    N = N_real if N_real is not None else N_phys
    QM = merged_indices.shape[2]
    C = H * D

    dq = torch.empty(B, N, H, D, dtype=q.dtype, device=q.device)

    _get_ext().sparse_attn_bwd_dq_launch(
        q, k, v, do_tensor,
        merged_indices, counts,
        softmax_max, softmax_lse, delta,
        dq,
        B, N, N_phys, H, D,
        QM, qblock_topk,
        q_block_size, kv_block_size,
        softmax_scale,
        q_pad_head, kv_pad_head,
    )

    return dq.view(B, N, C)


__all__ = ["flash_sparse_attn_bwd_dq_cuda"]
