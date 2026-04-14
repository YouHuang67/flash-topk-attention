"""CUDA CuTe MMA block scoring — drop-in replacement for Triton version."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

_CSRC_DIR = Path(__file__).parent / "csrc"
_CUTLASS_INCLUDE = "/home/hy/OpenRLHF/projects/flash-attention/csrc/cutlass/include"

_SUPPORTED_D = (32, 64, 96, 128, 160, 256)


@lru_cache(maxsize=None)
def _get_ext():
    cuda_src = (_CSRC_DIR / "block_score_kernel.cu").read_text()
    cpp_src = (_CSRC_DIR / "block_score.cpp").read_text()
    return load_inline(
        name="block_score_cuda",
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


def flash_block_score_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    num_heads: int,
    score_block_size: int,
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """CUDA CuTe MMA block scoring, drop-in for ``flash_block_score``.

    Args:
        q: ``[B, N, C]`` or ``[B, H, N, D]`` on CUDA (float16, bfloat16, or float32).
        k: Same shape/dtype as ``q``.
        num_heads: Head count.
        score_block_size: Tokens per score block (any positive integer).
        padding: ``(pad_head, pad_tail)``; padded length must divide
            ``score_block_size``.

    Returns:
        ``[B, H, N, M]`` float32 block scores summing to 1 per row.
    """
    if q.device.type != "cuda":
        raise ValueError("flash_block_score_cuda requires CUDA tensors")

    input_4d = q.ndim == 4
    if input_4d:
        B, H, N, D = q.shape
        if num_heads != H:
            raise ValueError(f"num_heads ({num_heads}) != q.shape[1] ({H})")
        C = H * D
    else:
        if q.ndim != 3:
            raise ValueError("q must be 3D [B, N, C] or 4D [B, H, N, D]")
        B, N, C = q.shape

    H = num_heads
    D = C // H

    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= score_block_size or pad_tail >= score_block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < score_block_size "
            f"({score_block_size})"
        )
    n_padded = pad_head + N + pad_tail
    if n_padded % score_block_size != 0:
        need = score_block_size - (n_padded % score_block_size)
        raise ValueError(
            f"pad_head + N + pad_tail = {n_padded} must be divisible by "
            f"score_block_size ({score_block_size}). "
            f"Try padding=(0, {pad_tail + need})."
        )

    if input_4d:
        if k.shape != (B, H, N, D):
            raise ValueError(f"k shape {k.shape} != {(B, H, N, D)}")
    else:
        if k.shape != (B, N, C):
            raise ValueError(f"k shape {k.shape} != {(B, N, C)}")
    if C % H != 0:
        raise ValueError(f"C={C} not divisible by num_heads={H}")
    if q.dtype != k.dtype:
        raise ValueError(f"q.dtype ({q.dtype}) != k.dtype ({k.dtype})")

    M = n_padded // score_block_size

    D_kernel = next(sd for sd in _SUPPORTED_D if sd >= D)
    pad_d = D_kernel - D

    is_fp32 = q.dtype == torch.float32
    if is_fp32:
        q_work = q.to(torch.bfloat16)
        k_work = k.to(torch.bfloat16)
    else:
        q_work = q
        k_work = k

    if input_4d:
        q_4d = q_work.contiguous()
        k_4d = k_work.contiguous()
    else:
        q_4d = q_work.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()
        k_4d = k_work.view(B, N, H, D).permute(0, 2, 1, 3).contiguous()

    if pad_d > 0:
        q_4d = F.pad(q_4d, (0, pad_d))
        k_4d = F.pad(k_4d, (0, pad_d))

    kBlockM = 64
    kBlockN_max = 64
    min_phys = max(kBlockM, kBlockN_max)
    N_phys = N
    if N_phys < min_phys:
        pad_n = min_phys - N_phys
        q_4d = F.pad(q_4d, (0, 0, 0, pad_n))
        k_4d = F.pad(k_4d, (0, 0, 0, pad_n))
        N_phys = min_phys

    m_locals = torch.empty(B * H, N, M, device=q.device, dtype=torch.float32)
    l_locals = torch.empty(B * H, N, M, device=q.device, dtype=torch.float32)
    block_scores = torch.empty(B * H, N, M, device=q.device, dtype=torch.float32)

    softmax_scale = D ** -0.5

    ext = _get_ext()
    ext.block_score_launch(
        q_4d, k_4d,
        m_locals, l_locals, block_scores,
        B, N, N_phys, H, D_kernel,
        M, score_block_size,
        softmax_scale,
        pad_head,
    )

    block_scores = block_scores.view(B, H, N, M)
    return block_scores


__all__ = ["flash_block_score_cuda"]
