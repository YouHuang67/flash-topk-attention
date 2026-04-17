"""JIT compilation wrapper for sparse attention CUDA kernel."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"
_CUTLASS_INCLUDE = "/home/hy/OpenRLHF/projects/flash-attention/csrc/cutlass/include"


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "sparse_attn_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "sparse_attn.cpp").read_text()
    return load_inline(
        name="sparse_attn_cuda",
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


def flash_sparse_attn_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    merged_indices: torch.Tensor,
    counts: torch.Tensor,
    num_heads: int,
    q_block_size: int,
    kv_block_size: int,
    qblock_topk: int,
    softmax_scale: float,
    q_pad_head: int,
    kv_pad_head: int,
    N_real: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch sparse attention CUDA kernel.

    Args:
        q: Query ``[B, N_phys, C]``, float16/bfloat16. N_phys >= N_real.
        k: Key ``[B, N_phys, C]``.
        v: Value ``[B, N_phys, C]``.
        merged_indices: ``[B, H, QM, qblock_topk]`` int32, -1 padding.
        counts: ``[B, H, QM]`` int32, valid count per qblock.
        num_heads: Number of attention heads.
        q_block_size: Query block size (1 to 64).
        kv_block_size: KV block size (any positive integer).
        qblock_topk: Max blocks per qblock.
        softmax_scale: Softmax scale factor.
        q_pad_head: Virtual Q head padding tokens.
        kv_pad_head: Virtual KV head padding tokens.
        N_real: Real sequence length (kernel uses this as N for masking).
            Defaults to q.shape[1].

    Returns:
        Tuple of (output, softmax_max, softmax_lse):
            - output: ``[B, N_real, C]``, same dtype as q.
            - softmax_max: ``[B, H, N_real]``, float32.
            - softmax_lse: ``[B, H, N_real]``, float32.
    """
    B, N_phys, C = q.shape
    H = num_heads
    D = C // H
    N = N_real if N_real is not None else N_phys
    QM = merged_indices.shape[2]

    q_bnhd = q.view(B, N_phys, H, D).contiguous()
    k_bnhd = k.view(B, N_phys, H, D).contiguous()
    v_bnhd = v.view(B, N_phys, H, D).contiguous()

    output = torch.empty(B, N, H, D, dtype=q.dtype, device=q.device)
    softmax_max = torch.empty(B, H, N, dtype=torch.float32, device=q.device)
    softmax_lse = torch.empty(B, H, N, dtype=torch.float32, device=q.device)

    merged_indices = merged_indices.contiguous()
    counts = counts.contiguous()

    _get_ext().sparse_attn_fwd_launch(
        q_bnhd, k_bnhd, v_bnhd,
        merged_indices, counts,
        output, softmax_max, softmax_lse,
        B, N, N_phys, H, D,
        QM, qblock_topk,
        q_block_size, kv_block_size,
        softmax_scale,
        q_pad_head, kv_pad_head,
    )

    output = output.view(B, N, C)
    return output, softmax_max, softmax_lse


__all__ = ["flash_sparse_attn_cuda"]
