import math
import warnings
from typing import Optional, Tuple

from einops import rearrange

import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _compare_and_swap(x, ids, flip, i: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2**i, 2, 2**(n_dims - i - 1)]
    y = tl.reshape(x, shape)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)

    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)

    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(x, ids, stage: tl.constexpr, order: tl.constexpr, n_dims: tl.constexpr):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2**(n_dims - 1 - stage), 2, 2**stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def _bitonic_sort_pool(pool_scores, pool_indices, n_dims_merge: tl.constexpr):
    """Full bitonic sort: build bitonic sequence then merge descending."""
    for stage in tl.static_range(1, n_dims_merge):
        pool_scores, pool_indices = _bitonic_merge(pool_scores, pool_indices, stage, 2, n_dims_merge)
    pool_scores, pool_indices = _bitonic_merge(
        pool_scores, pool_indices, n_dims_merge, True, n_dims_merge
    )
    return pool_scores, pool_indices


MAX_KV_BS_IN_CONFIGS = 128
MAX_ALLOWED_ENTRIES_PER_KV = 16


@triton.autotune(
    configs=[
        triton.Config({'Q_BS': q, 'KV_BS': kv}, num_warps=4 if q < 64 else 8)
        for q in [16, 32, 64] for kv in [16, 32, 64, 128]
    ],
    key=['N', 'N_PADDED', 'SCORE_K', 'SCORE_BS', 'SCORE_BS_ORIG', 'IS_POW2'],
)
@triton.jit
def flash_scoring_kernel(
    Q, K, V, O, LSE, TopK_Indices, TopK_Scores,
    B: tl.constexpr, N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    Q_BS: tl.constexpr, KV_BS: tl.constexpr,
    IS_POW2: tl.constexpr,
    SCORE_BS: tl.constexpr, SCORE_BS_ORIG: tl.constexpr,
    SCORE_K: tl.constexpr, SCORE_POOL: tl.constexpr,
    PAD_HEAD: tl.constexpr, PAD_TAIL: tl.constexpr, N_PADDED: tl.constexpr,
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    stride_lse_b, stride_lse_h, stride_lse_n,
    stride_tki_b, stride_tki_h, stride_tki_n, stride_tki_k,
    stride_tks_b, stride_tks_h, stride_tks_n, stride_tks_k,
):
    """Flash Attention kernel with TopK block scoring.

    Supports non-power-of-2 SCORE_BS_ORIG via compile-time branching.
    """
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)

    q_start = i_t * Q_BS
    scale = (1.0 / math.sqrt(D)) * 1.44269504
    NEG_INF: tl.constexpr = -1e38

    p_q = tl.make_block_ptr(
        base=Q + i_b * stride_q_b + i_h * stride_q_h,
        shape=(N, D), strides=(stride_q_n, stride_q_d),
        offsets=(q_start, 0), block_shape=(Q_BS, D), order=(1, 0),
    )
    q = tl.load(p_q, boundary_check=(0, 1))

    o_acc = tl.zeros((Q_BS, D), dtype=tl.float32)
    m_global = tl.full((Q_BS,), float('-inf'), dtype=tl.float32)
    l_global = tl.zeros((Q_BS,), dtype=tl.float32)

    pool_scores = tl.full((Q_BS, SCORE_POOL), NEG_INF, dtype=tl.float32)
    pool_indices = tl.full((Q_BS, SCORE_POOL), -1, dtype=tl.int32)

    HALF_POOL: tl.constexpr = SCORE_POOL // 2
    n_dims_merge: tl.constexpr = int(math.log2(SCORE_POOL))

    if IS_POW2:
        num_kv_blocks: tl.constexpr = (N_PADDED + KV_BS - 1) // KV_BS
        num_score_blocks: tl.constexpr = N_PADDED // SCORE_BS
        inv_count_first = 1.0 / float(SCORE_BS - PAD_HEAD)
        inv_count_last = 1.0 / float(SCORE_BS - PAD_TAIL)
        inv_count_mid = 1.0 / float(SCORE_BS)

        p_k = tl.make_block_ptr(
            base=K + i_b * stride_k_b + i_h * stride_k_h,
            shape=(D, N), strides=(stride_k_d, stride_k_n),
            offsets=(0, -PAD_HEAD), block_shape=(D, KV_BS), order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=V + i_b * stride_v_b + i_h * stride_v_h,
            shape=(N, D), strides=(stride_v_n, stride_v_d),
            offsets=(-PAD_HEAD, 0), block_shape=(KV_BS, D), order=(1, 0),
        )

        for i_block in range(num_kv_blocks):
            k = tl.load(p_k, boundary_check=(0, 1))
            s = tl.dot(q, k)
            kv_idx = tl.arange(0, KV_BS)
            if PAD_HEAD > 0 or PAD_TAIL > 0:
                real_pos = i_block * KV_BS + kv_idx - PAD_HEAD
                s = tl.where((real_pos[None, :] >= 0) & (real_pos[None, :] < N), s, NEG_INF)
            else:
                s = tl.where((i_block * KV_BS + kv_idx)[None, :] < N, s, NEG_INF)

            m_local = tl.max(s, axis=1)
            m_prev = m_global
            m_global = tl.maximum(m_global, m_local)

            alpha = tl.where(m_prev == float('-inf'), 1.0, tl.exp2((m_prev - m_global) * scale))
            o_acc = o_acc * alpha[:, None]
            l_global = l_global * alpha
            pool_scores = pool_scores * alpha[:, None]

            p = tl.exp2((s - m_global[:, None]) * scale)
            l_local = tl.sum(p, axis=1)
            l_global = l_global + l_local

            v = tl.load(p_v, boundary_check=(0, 1))
            o_acc = o_acc + tl.dot(p.to(v.dtype), v)

            p_k = tl.advance(p_k, (0, KV_BS))
            p_v = tl.advance(p_v, (KV_BS, 0))

            if KV_BS > SCORE_BS:
                ENTRIES_PER_KV: tl.constexpr = KV_BS // SCORE_BS
                p_3d = tl.reshape(p, (Q_BS, ENTRIES_PER_KV, SCORE_BS))
                l_sub = tl.sum(p_3d, axis=2)

                for j in tl.static_range(ENTRIES_PER_KV):
                    cur_entry = i_block * ENTRIES_PER_KV + j
                    pool_pos_j = tl.where(cur_entry < HALF_POOL, cur_entry,
                                          HALF_POOL + (cur_entry % HALF_POOL))
                    mask_j = tl.arange(0, SCORE_POOL) == pool_pos_j

                    mask_entry = tl.arange(0, ENTRIES_PER_KV) == j
                    l_j = tl.sum(tl.where(mask_entry[None, :], l_sub, 0.0), axis=1)
                    if PAD_HEAD > 0 or PAD_TAIL > 0:
                        inv_c = tl.where(
                            cur_entry == 0,
                            inv_count_first,
                            tl.where(cur_entry == num_score_blocks - 1, inv_count_last, inv_count_mid),
                        )
                        l_j = l_j * inv_c

                    pool_scores = tl.where(mask_j[None, :], l_j[:, None], pool_scores)
                    pool_indices = tl.where(mask_j[None, :], cur_entry, pool_indices)

                total_entries = (i_block + 1) * ENTRIES_PER_KV
                if (total_entries >= HALF_POOL) and (total_entries % HALF_POOL == 0):
                    pool_scores, pool_indices = _bitonic_sort_pool(
                        pool_scores, pool_indices, n_dims_merge
                    )

            elif KV_BS == SCORE_BS:
                pool_base = tl.where(i_block >= HALF_POOL, HALF_POOL + (i_block % HALF_POOL), i_block)
                mask_pool_base = tl.arange(0, SCORE_POOL) == pool_base
                l_block = l_local
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    inv_c = tl.where(
                        i_block == 0,
                        inv_count_first,
                        tl.where(i_block == num_score_blocks - 1, inv_count_last, inv_count_mid),
                    )
                    l_block = l_block * inv_c
                pool_scores = tl.where(mask_pool_base[None, :], l_block[:, None], pool_scores)
                pool_indices = tl.where(mask_pool_base[None, :], i_block, pool_indices)

                if ((i_block + 1) >= HALF_POOL) and ((i_block + 1) % HALF_POOL == 0):
                    pool_scores, pool_indices = _bitonic_sort_pool(
                        pool_scores, pool_indices, n_dims_merge
                    )

            else:
                KV_ITERS_PER_ENTRY: tl.constexpr = SCORE_BS // KV_BS
                sub_iter = i_block & (KV_ITERS_PER_ENTRY - 1)
                score_block_idx = i_block // KV_ITERS_PER_ENTRY

                pool_pos = tl.where(score_block_idx < HALF_POOL, score_block_idx,
                                   HALF_POOL + (score_block_idx % HALF_POOL))
                mask_pos = tl.arange(0, SCORE_POOL) == pool_pos

                old_val = tl.where(sub_iter == 0, 0.0, pool_scores)
                l_add = l_local
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    inv_c = tl.where(
                        score_block_idx == 0,
                        inv_count_first,
                        tl.where(score_block_idx == num_score_blocks - 1, inv_count_last, inv_count_mid),
                    )
                    l_add = l_add * inv_c
                pool_scores = tl.where(mask_pos[None, :],
                                      old_val + l_add[:, None],
                                      pool_scores)

                pool_indices = tl.where(mask_pos[None, :] & (sub_iter == 0),
                                       score_block_idx, pool_indices)

                if sub_iter == KV_ITERS_PER_ENTRY - 1:
                    total_entries = score_block_idx + 1
                    if (total_entries >= HALF_POOL) and (total_entries % HALF_POOL == 0):
                        pool_scores, pool_indices = _bitonic_sort_pool(
                            pool_scores, pool_indices, n_dims_merge
                        )

        if KV_BS > SCORE_BS:
            ENTRIES_PER_KV: tl.constexpr = KV_BS // SCORE_BS
            total_entries = num_kv_blocks * ENTRIES_PER_KV
            if (total_entries % HALF_POOL) != 0:
                pool_scores, pool_indices = _bitonic_sort_pool(
                    pool_scores, pool_indices, n_dims_merge
                )
        elif KV_BS == SCORE_BS:
            if (num_kv_blocks % HALF_POOL) != 0:
                pool_scores, pool_indices = _bitonic_sort_pool(
                    pool_scores, pool_indices, n_dims_merge
                )
        else:
            KV_ITERS_PER_ENTRY: tl.constexpr = SCORE_BS // KV_BS
            total_score_blocks = (num_kv_blocks + KV_ITERS_PER_ENTRY - 1) // KV_ITERS_PER_ENTRY
            if (total_score_blocks % HALF_POOL) != 0:
                pool_scores, pool_indices = _bitonic_sort_pool(
                    pool_scores, pool_indices, n_dims_merge
                )

    else:
        CASE_LARGE: tl.constexpr = SCORE_BS_ORIG >= KV_BS

        if CASE_LARGE:
            KV_ITERS_PER_BLOCK: tl.constexpr = (SCORE_BS_ORIG + KV_BS - 1) // KV_BS
            num_score_blocks: tl.constexpr = N_PADDED // SCORE_BS_ORIG
            num_kv_iters: tl.constexpr = num_score_blocks * KV_ITERS_PER_BLOCK
            inv_count_first = 1.0 / float(SCORE_BS_ORIG - PAD_HEAD)
            inv_count_last = 1.0 / float(SCORE_BS_ORIG - PAD_TAIL)
            inv_count_mid = 1.0 / float(SCORE_BS_ORIG)

            k_base = K + i_b * stride_k_b + i_h * stride_k_h
            v_base = V + i_b * stride_v_b + i_h * stride_v_h

            for i_kv in range(num_kv_iters):
                block_idx = i_kv // KV_ITERS_PER_BLOCK
                sub_iter = i_kv % KV_ITERS_PER_BLOCK
                block_base = block_idx * SCORE_BS_ORIG - PAD_HEAD
                local_off = sub_iter * KV_BS
                global_off = block_base + local_off

                p_k = tl.make_block_ptr(
                    base=k_base,
                    shape=(D, N), strides=(stride_k_d, stride_k_n),
                    offsets=(0, global_off),
                    block_shape=(D, KV_BS), order=(0, 1),
                )
                k = tl.load(p_k, boundary_check=(0, 1))
                s = tl.dot(q, k)

                kv_idx = tl.arange(0, KV_BS)
                intra_valid = (local_off + kv_idx) < SCORE_BS_ORIG
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    kv_global = global_off + kv_idx
                    valid_mask = intra_valid & (kv_global >= 0) & (kv_global < N)
                else:
                    valid_mask = intra_valid
                s = tl.where(valid_mask[None, :], s, NEG_INF)

                m_local = tl.max(s, axis=1)
                m_prev = m_global
                m_global = tl.maximum(m_global, m_local)

                alpha = tl.where(m_prev == float('-inf'), 1.0, tl.exp2((m_prev - m_global) * scale))
                o_acc = o_acc * alpha[:, None]
                l_global = l_global * alpha
                pool_scores = pool_scores * alpha[:, None]

                p = tl.exp2((s - m_global[:, None]) * scale)
                l_local = tl.sum(p, axis=1)
                l_global = l_global + l_local

                p_v = tl.make_block_ptr(
                    base=v_base,
                    shape=(N, D), strides=(stride_v_n, stride_v_d),
                    offsets=(global_off, 0),
                    block_shape=(KV_BS, D), order=(1, 0),
                )
                v = tl.load(p_v, boundary_check=(0, 1))
                o_acc = o_acc + tl.dot(p.to(v.dtype), v)

                pool_pos = tl.where(block_idx < HALF_POOL, block_idx,
                                   HALF_POOL + (block_idx % HALF_POOL))
                mask_pos = tl.arange(0, SCORE_POOL) == pool_pos

                old_val = tl.where(sub_iter == 0, 0.0, pool_scores)
                l_add = l_local
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    inv_c = tl.where(
                        block_idx == 0,
                        inv_count_first,
                        tl.where(block_idx == num_score_blocks - 1, inv_count_last, inv_count_mid),
                    )
                    l_add = l_add * inv_c
                pool_scores = tl.where(mask_pos[None, :],
                                      old_val + l_add[:, None], pool_scores)
                pool_indices = tl.where(mask_pos[None, :] & (sub_iter == 0),
                                       block_idx, pool_indices)

                if sub_iter == KV_ITERS_PER_BLOCK - 1:
                    total_score_blocks_so_far = block_idx + 1
                    if (total_score_blocks_so_far >= HALF_POOL) and (total_score_blocks_so_far % HALF_POOL == 0):
                        pool_scores, pool_indices = _bitonic_sort_pool(
                            pool_scores, pool_indices, n_dims_merge
                        )

            if (num_score_blocks % HALF_POOL) != 0:
                pool_scores, pool_indices = _bitonic_sort_pool(
                    pool_scores, pool_indices, n_dims_merge
                )

        else:
            ENTRIES_PER_KV: tl.constexpr = KV_BS // SCORE_BS
            EFFECTIVE_KV_LEN: tl.constexpr = ENTRIES_PER_KV * SCORE_BS_ORIG
            num_kv_iters: tl.constexpr = (N_PADDED + EFFECTIVE_KV_LEN - 1) // EFFECTIVE_KV_LEN
            num_score_blocks: tl.constexpr = N_PADDED // SCORE_BS_ORIG
            inv_count_first = 1.0 / float(SCORE_BS_ORIG - PAD_HEAD)
            inv_count_last = 1.0 / float(SCORE_BS_ORIG - PAD_TAIL)
            inv_count_mid = 1.0 / float(SCORE_BS_ORIG)

            k_base = K + i_b * stride_k_b + i_h * stride_k_h
            v_base = V + i_b * stride_v_b + i_h * stride_v_h

            j_out = tl.arange(0, KV_BS)
            entry_id = j_out // SCORE_BS
            pos_in_entry = j_out % SCORE_BS
            valid_load = pos_in_entry < SCORE_BS_ORIG

            for i_kv in range(num_kv_iters):
                kv_base = i_kv * EFFECTIVE_KV_LEN
                j_in = kv_base + entry_id * SCORE_BS_ORIG + pos_in_entry - PAD_HEAD

                d_idx = tl.arange(0, D)[:, None]
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    load_mask = valid_load[None, :] & (j_in[None, :] >= 0) & (j_in[None, :] < N)
                else:
                    load_mask = valid_load[None, :] & (j_in[None, :] < N)
                k = tl.load(
                    k_base + d_idx * stride_k_d + j_in[None, :] * stride_k_n,
                    mask=load_mask,
                    other=0.0
                )

                s = tl.dot(q, k)
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    s = tl.where(load_mask, s, NEG_INF)
                else:
                    s = tl.where(valid_load[None, :] & (j_in[None, :] < N), s, NEG_INF)

                m_local = tl.max(s, axis=1)
                m_prev = m_global
                m_global = tl.maximum(m_global, m_local)

                alpha = tl.where(m_prev == float('-inf'), 1.0, tl.exp2((m_prev - m_global) * scale))
                o_acc = o_acc * alpha[:, None]
                l_global = l_global * alpha
                pool_scores = pool_scores * alpha[:, None]

                p = tl.exp2((s - m_global[:, None]) * scale)
                l_local = tl.sum(p, axis=1)
                l_global = l_global + l_local

                d_idx_v = tl.arange(0, D)
                if PAD_HEAD > 0 or PAD_TAIL > 0:
                    v_mask = valid_load[:, None] & (j_in[:, None] >= 0) & (j_in[:, None] < N)
                else:
                    v_mask = valid_load[:, None] & (j_in[:, None] < N)
                v = tl.load(
                    v_base + j_in[:, None] * stride_v_n + d_idx_v[None, :] * stride_v_d,
                    mask=v_mask,
                    other=0.0
                )
                o_acc = o_acc + tl.dot(p.to(v.dtype), v)

                p_3d = tl.reshape(p, (Q_BS, ENTRIES_PER_KV, SCORE_BS))
                l_sub = tl.sum(p_3d, axis=2)

                for j in tl.static_range(ENTRIES_PER_KV):
                    cur_entry = i_kv * ENTRIES_PER_KV + j
                    pool_pos_j = tl.where(cur_entry < HALF_POOL, cur_entry,
                                          HALF_POOL + (cur_entry % HALF_POOL))
                    mask_j = tl.arange(0, SCORE_POOL) == pool_pos_j

                    mask_entry = tl.arange(0, ENTRIES_PER_KV) == j
                    l_j = tl.sum(tl.where(mask_entry[None, :], l_sub, 0.0), axis=1)
                    if PAD_HEAD > 0 or PAD_TAIL > 0:
                        inv_c = tl.where(
                            cur_entry == 0,
                            inv_count_first,
                            tl.where(cur_entry == num_score_blocks - 1, inv_count_last, inv_count_mid),
                        )
                        l_j = l_j * inv_c

                    pool_scores = tl.where(mask_j[None, :], l_j[:, None], pool_scores)
                    pool_indices = tl.where(mask_j[None, :], cur_entry, pool_indices)

                total_entries = (i_kv + 1) * ENTRIES_PER_KV
                if (total_entries >= HALF_POOL) and (total_entries % HALF_POOL == 0):
                    pool_scores, pool_indices = _bitonic_sort_pool(
                        pool_scores, pool_indices, n_dims_merge
                    )

            total_entries = num_kv_iters * ENTRIES_PER_KV
            if (total_entries % HALF_POOL) != 0:
                pool_scores, pool_indices = _bitonic_sort_pool(
                    pool_scores, pool_indices, n_dims_merge
                )

    o = o_acc / l_global[:, None]

    p_o = tl.make_block_ptr(
        base=O + i_b * stride_o_b + i_h * stride_o_h,
        shape=(N, D), strides=(stride_o_n, stride_o_d),
        offsets=(q_start, 0), block_shape=(Q_BS, D), order=(1, 0),
    )
    tl.store(p_o, o.to(O.dtype.element_ty), boundary_check=(0, 1))

    lse = m_global / math.sqrt(D) + tl.log(l_global)
    p_lse = tl.make_block_ptr(
        base=LSE + i_b * stride_lse_b + i_h * stride_lse_h,
        shape=(N,), strides=(stride_lse_n,),
        offsets=(q_start,), block_shape=(Q_BS,), order=(0,),
    )
    tl.store(p_lse, lse, boundary_check=(0,))

    pool_scores = tl.where(pool_scores == NEG_INF, 0.0, pool_scores)
    final_scores = pool_scores / l_global[:, None]

    p_scores = tl.make_block_ptr(
        base=TopK_Scores + i_b * stride_tks_b + i_h * stride_tks_h,
        shape=(N, SCORE_POOL), strides=(stride_tks_n, stride_tks_k),
        offsets=(q_start, 0), block_shape=(Q_BS, SCORE_POOL), order=(1, 0),
    )
    tl.store(p_scores, final_scores.to(TopK_Scores.dtype.element_ty), boundary_check=(0,))

    p_indices = tl.make_block_ptr(
        base=TopK_Indices + i_b * stride_tki_b + i_h * stride_tki_h,
        shape=(N, SCORE_POOL), strides=(stride_tki_n, stride_tki_k),
        offsets=(q_start, 0), block_shape=(Q_BS, SCORE_POOL), order=(1, 0),
    )
    tl.store(p_indices, pool_indices.to(TopK_Indices.dtype.element_ty), boundary_check=(0,))


@triton.jit
def flash_scoring_delta_kernel(
    O, DO, Delta,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    N: tl.constexpr, H: tl.constexpr, D: tl.constexpr, BD: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)

    p_o = tl.make_block_ptr(
        base=O + i_b * stride_o_b + i_n * stride_o_n + i_h * stride_o_h,
        shape=(D,), strides=(stride_o_d,),
        offsets=(0,), block_shape=(BD,), order=(0,),
    )
    p_do = tl.make_block_ptr(
        base=DO + i_b * stride_do_b + i_n * stride_do_n + i_h * stride_do_h,
        shape=(D,), strides=(stride_do_d,),
        offsets=(0,), block_shape=(BD,), order=(0,),
    )
    b_o = tl.load(p_o, boundary_check=(0,))
    b_do = tl.load(p_do, boundary_check=(0,))
    delta_val = tl.sum(b_o * b_do)
    tl.store(Delta + i_b * H * N + i_h * N + i_n, delta_val.to(Delta.dtype.element_ty))


@triton.autotune(
    configs=[
        triton.Config({'Q_BS': q, 'KV_BS': kv}, num_warps=4 if q < 64 else 8)
        for q in [16, 32, 64] for kv in [16, 32, 64, 128]
    ],
    key=['N', 'D'],
)
@triton.jit
def flash_scoring_dq_kernel(
    Q, K, V, DO, DQ, LSE, Delta, scale,
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_dq_b, stride_dq_n, stride_dq_h, stride_dq_d,
    N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    Q_BS: tl.constexpr, KV_BS: tl.constexpr,
):
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)

    q_start = i_t * Q_BS

    p_q = tl.make_block_ptr(
        base=Q + i_b * stride_q_b + i_h * stride_q_h,
        shape=(N, D), strides=(stride_q_n, stride_q_d),
        offsets=(q_start, 0), block_shape=(Q_BS, D), order=(1, 0),
    )
    p_do = tl.make_block_ptr(
        base=DO + i_b * stride_do_b + i_h * stride_do_h,
        shape=(N, D), strides=(stride_do_n, stride_do_d),
        offsets=(q_start, 0), block_shape=(Q_BS, D), order=(1, 0),
    )
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_do = tl.load(p_do, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_lse = tl.make_block_ptr(
        base=LSE + i_b * H * N + i_h * N,
        shape=(N,), strides=(1,),
        offsets=(q_start,), block_shape=(Q_BS,), order=(0,),
    )
    p_delta = tl.make_block_ptr(
        base=Delta + i_b * H * N + i_h * N,
        shape=(N,), strides=(1,),
        offsets=(q_start,), block_shape=(Q_BS,), order=(0,),
    )
    b_lse = tl.load(p_lse, boundary_check=(0,))
    b_delta = tl.load(p_delta, boundary_check=(0,))

    b_dq = tl.zeros([Q_BS, D], dtype=tl.float32)

    for i_c in range(0, N, KV_BS):
        p_k = tl.make_block_ptr(
            base=K + i_b * stride_k_b + i_h * stride_k_h,
            shape=(D, N), strides=(stride_k_d, stride_k_n),
            offsets=(0, i_c), block_shape=(D, KV_BS), order=(0, 1),
        )
        p_v = tl.make_block_ptr(
            base=V + i_b * stride_v_b + i_h * stride_v_h,
            shape=(N, D), strides=(stride_v_n, stride_v_d),
            offsets=(i_c, 0), block_shape=(KV_BS, D), order=(1, 0),
        )
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        b_s = tl.dot(b_q, b_k)
        b_p = tl.exp(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_dp - b_delta[:, None])
        b_dq += tl.dot(b_ds.to(b_k.dtype), tl.trans(b_k))

    b_dq = b_dq * scale

    p_dq = tl.make_block_ptr(
        base=DQ + i_b * stride_dq_b + i_h * stride_dq_h,
        shape=(N, D), strides=(stride_dq_n, stride_dq_d),
        offsets=(q_start, 0), block_shape=(Q_BS, D), order=(1, 0),
    )
    tl.store(p_dq, b_dq.to(DQ.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({'KV_BS': kv, 'Q_BS': q}, num_warps=4 if kv < 64 else 8)
        for kv in [16, 32, 64, 128] for q in [16, 32, 64]
    ],
    key=['N', 'D'],
)
@triton.jit
def flash_scoring_dkv_kernel(
    Q, K, V, DO, DK, DV, LSE, Delta, scale,
    stride_q_b, stride_q_n, stride_q_h, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    Q_BS: tl.constexpr, KV_BS: tl.constexpr,
):
    i_c = tl.program_id(0)
    i_h = tl.program_id(1)
    i_b = tl.program_id(2)

    kv_start = i_c * KV_BS

    p_k = tl.make_block_ptr(
        base=K + i_b * stride_k_b + i_h * stride_k_h,
        shape=(N, D), strides=(stride_k_n, stride_k_d),
        offsets=(kv_start, 0), block_shape=(KV_BS, D), order=(1, 0),
    )
    p_v = tl.make_block_ptr(
        base=V + i_b * stride_v_b + i_h * stride_v_h,
        shape=(N, D), strides=(stride_v_n, stride_v_d),
        offsets=(kv_start, 0), block_shape=(KV_BS, D), order=(1, 0),
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    b_dk = tl.zeros([KV_BS, D], dtype=tl.float32)
    b_dv = tl.zeros([KV_BS, D], dtype=tl.float32)

    for i_t in range(0, N, Q_BS):
        p_q = tl.make_block_ptr(
            base=Q + i_b * stride_q_b + i_h * stride_q_h,
            shape=(N, D), strides=(stride_q_n, stride_q_d),
            offsets=(i_t, 0), block_shape=(Q_BS, D), order=(1, 0),
        )
        p_do = tl.make_block_ptr(
            base=DO + i_b * stride_do_b + i_h * stride_do_h,
            shape=(N, D), strides=(stride_do_n, stride_do_d),
            offsets=(i_t, 0), block_shape=(Q_BS, D), order=(1, 0),
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_do = tl.load(p_do, boundary_check=(0, 1))
        b_q = (b_q * scale).to(b_q.dtype)

        p_lse = tl.make_block_ptr(
            base=LSE + i_b * H * N + i_h * N,
            shape=(N,), strides=(1,),
            offsets=(i_t,), block_shape=(Q_BS,), order=(0,),
        )
        p_delta = tl.make_block_ptr(
            base=Delta + i_b * H * N + i_h * N,
            shape=(N,), strides=(1,),
            offsets=(i_t,), block_shape=(Q_BS,), order=(0,),
        )
        b_lse = tl.load(p_lse, boundary_check=(0,))
        b_delta = tl.load(p_delta, boundary_check=(0,))

        b_s = tl.dot(b_q, tl.trans(b_k))
        b_p = tl.exp(b_s - b_lse[:, None])
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_dp - b_delta[:, None])
        b_dk += tl.dot(tl.trans(b_ds.to(b_q.dtype)), b_q)
        b_dv += tl.dot(tl.trans(b_p.to(b_do.dtype)), b_do)

    p_dk = tl.make_block_ptr(
        base=DK + i_b * stride_dk_b + i_h * stride_dk_h,
        shape=(N, D), strides=(stride_dk_n, stride_dk_d),
        offsets=(kv_start, 0), block_shape=(KV_BS, D), order=(1, 0),
    )
    p_dv = tl.make_block_ptr(
        base=DV + i_b * stride_dv_b + i_h * stride_dv_h,
        shape=(N, D), strides=(stride_dv_n, stride_dv_d),
        offsets=(kv_start, 0), block_shape=(KV_BS, D), order=(1, 0),
    )
    tl.store(p_dk, b_dk.to(DK.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(DV.dtype.element_ty), boundary_check=(0, 1))


class FlashScoringFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, num_heads, score_block_size, topk, padding: Tuple[int, int]):
        B, N, C = q.shape
        H = num_heads
        D = C // H

        assert k.shape == (B, N, C)
        assert v.shape == (B, N, C)
        assert C % H == 0
        assert score_block_size >= 1

        SCORE_K = _next_power_of_2(topk)
        SCORE_BS_ORIG = score_block_size
        SCORE_BS = _next_power_of_2(SCORE_BS_ORIG)
        IS_POW2 = (SCORE_BS_ORIG & (SCORE_BS_ORIG - 1)) == 0

        pad_head, pad_tail = padding
        if pad_head < 0 or pad_tail < 0:
            raise ValueError("padding must be non-negative (pad_head, pad_tail)")
        if pad_head >= SCORE_BS_ORIG or pad_tail >= SCORE_BS_ORIG:
            raise ValueError(
                f"padding ({pad_head}, {pad_tail}) must each be < score_block_size ({SCORE_BS_ORIG})"
            )
        n_padded = pad_head + N + pad_tail

        if n_padded % SCORE_BS_ORIG != 0:
            need = SCORE_BS_ORIG - (n_padded % SCORE_BS_ORIG)
            raise ValueError(
                f"pad_head + N + pad_tail = {n_padded} must be divisible by "
                f"score_block_size ({SCORE_BS_ORIG}). Try padding=(0, {pad_tail + need})."
            )
        num_blocks = n_padded // SCORE_BS_ORIG
        if topk > num_blocks:
            raise ValueError(f"topk ({topk}) > num_blocks ({num_blocks})")

        max_valid_kv_bs = min(SCORE_BS * MAX_ALLOWED_ENTRIES_PER_KV, MAX_KV_BS_IN_CONFIGS)
        max_entries_per_kv = max_valid_kv_bs // SCORE_BS
        SCORE_POOL = max(2 * SCORE_K, 2 * max_entries_per_kv)
        SCORE_POOL = 2 ** math.ceil(math.log2(SCORE_POOL))

        q_4d = q.reshape(B, N, H, D).contiguous()
        k_4d = k.reshape(B, N, H, D).contiguous()
        v_4d = v.reshape(B, N, H, D).contiguous()

        O = torch.empty(B, N, H, D, device=q.device, dtype=q.dtype)
        LSE = torch.empty(B, H, N, device=q.device, dtype=torch.float32)
        Pool_Indices = torch.empty(B, H, N, SCORE_POOL, device=q.device, dtype=torch.int32)
        Pool_Scores = torch.empty(B, H, N, SCORE_POOL, device=q.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(N, meta['Q_BS']), H, B)
        flash_scoring_kernel[grid](
            q_4d, k_4d, v_4d, O, LSE, Pool_Indices, Pool_Scores,
            B=B, N=N, H=H, D=D,
            IS_POW2=IS_POW2, SCORE_BS=SCORE_BS, SCORE_BS_ORIG=SCORE_BS_ORIG,
            SCORE_K=SCORE_K, SCORE_POOL=SCORE_POOL,
            PAD_HEAD=pad_head, PAD_TAIL=pad_tail, N_PADDED=n_padded,
            stride_q_b=q_4d.stride(0), stride_q_n=q_4d.stride(1),
            stride_q_h=q_4d.stride(2), stride_q_d=q_4d.stride(3),
            stride_k_b=k_4d.stride(0), stride_k_n=k_4d.stride(1),
            stride_k_h=k_4d.stride(2), stride_k_d=k_4d.stride(3),
            stride_v_b=v_4d.stride(0), stride_v_n=v_4d.stride(1),
            stride_v_h=v_4d.stride(2), stride_v_d=v_4d.stride(3),
            stride_o_b=O.stride(0), stride_o_n=O.stride(1),
            stride_o_h=O.stride(2), stride_o_d=O.stride(3),
            stride_lse_b=LSE.stride(0), stride_lse_h=LSE.stride(1), stride_lse_n=LSE.stride(2),
            stride_tki_b=Pool_Indices.stride(0), stride_tki_h=Pool_Indices.stride(1),
            stride_tki_n=Pool_Indices.stride(2), stride_tki_k=Pool_Indices.stride(3),
            stride_tks_b=Pool_Scores.stride(0), stride_tks_h=Pool_Scores.stride(1),
            stride_tks_n=Pool_Scores.stride(2), stride_tks_k=Pool_Scores.stride(3),
        )

        effective_k = min(topk, num_blocks)
        TopK_Scores = Pool_Scores[..., :effective_k]
        TopK_Indices = Pool_Indices[..., :effective_k]

        ctx.save_for_backward(q_4d, k_4d, v_4d, O, LSE)
        ctx.num_heads = H
        ctx.score_block_size = SCORE_BS_ORIG

        return O.reshape(B, N, C).to(q.dtype), TopK_Indices, TopK_Scores

    @staticmethod
    def backward(ctx, grad_o, grad_topk_indices, grad_topk_scores):
        if grad_topk_scores is not None and grad_topk_scores.abs().sum() > 0:
            warnings.warn("topk_scores gradients are not implemented, ignoring...", UserWarning)

        q_4d, k_4d, v_4d, O, LSE = ctx.saved_tensors
        H = ctx.num_heads
        B, N, C = grad_o.shape
        D = C // H
        scale = D ** -0.5

        grad_o_4d = grad_o.reshape(B, N, H, D).contiguous()
        BD = min(128, triton.next_power_of_2(D))

        Delta = torch.empty(B, H, N, device=grad_o.device, dtype=torch.float32)
        flash_scoring_delta_kernel[(N, H, B)](
            O, grad_o_4d, Delta,
            O.stride(0), O.stride(1), O.stride(2), O.stride(3),
            grad_o_4d.stride(0), grad_o_4d.stride(1), grad_o_4d.stride(2), grad_o_4d.stride(3),
            N=N, H=H, D=D, BD=BD,
        )

        DQ = torch.zeros(B, N, H, D, device=grad_o.device, dtype=grad_o.dtype)
        grid_dq = lambda meta: (triton.cdiv(N, meta['Q_BS']), H, B)
        flash_scoring_dq_kernel[grid_dq](
            q_4d, k_4d, v_4d, grad_o_4d, DQ, LSE, Delta, scale,
            q_4d.stride(0), q_4d.stride(1), q_4d.stride(2), q_4d.stride(3),
            k_4d.stride(0), k_4d.stride(1), k_4d.stride(2), k_4d.stride(3),
            v_4d.stride(0), v_4d.stride(1), v_4d.stride(2), v_4d.stride(3),
            grad_o_4d.stride(0), grad_o_4d.stride(1), grad_o_4d.stride(2), grad_o_4d.stride(3),
            DQ.stride(0), DQ.stride(1), DQ.stride(2), DQ.stride(3),
            N=N, H=H, D=D,
        )

        DK = torch.zeros(B, N, H, D, device=grad_o.device, dtype=grad_o.dtype)
        DV = torch.zeros(B, N, H, D, device=grad_o.device, dtype=grad_o.dtype)
        grid_dkv = lambda meta: (triton.cdiv(N, meta['KV_BS']), H, B)
        flash_scoring_dkv_kernel[grid_dkv](
            q_4d, k_4d, v_4d, grad_o_4d, DK, DV, LSE, Delta, scale,
            q_4d.stride(0), q_4d.stride(1), q_4d.stride(2), q_4d.stride(3),
            k_4d.stride(0), k_4d.stride(1), k_4d.stride(2), k_4d.stride(3),
            v_4d.stride(0), v_4d.stride(1), v_4d.stride(2), v_4d.stride(3),
            grad_o_4d.stride(0), grad_o_4d.stride(1), grad_o_4d.stride(2), grad_o_4d.stride(3),
            DK.stride(0), DK.stride(1), DK.stride(2), DK.stride(3),
            DV.stride(0), DV.stride(1), DV.stride(2), DV.stride(3),
            N=N, H=H, D=D,
        )

        return DQ.reshape(B, N, C), DK.reshape(B, N, C), DV.reshape(B, N, C), None, None, None, None


def flash_topk_score(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: Optional[int] = None,
    score_block_size: int = 64,
    topk: int = 16,
    padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash Attention with TopK block scoring (Triton kernel).

    Args:
        q: Query [B, N, C] or [B, H, N, D] with C = num_heads * D.
        k, v: Same layout as q.
        num_heads: Required for 3D input; optional for 4D (inferred from shape[1]).
        score_block_size: Scoring block size (any positive integer).
            Internally padded to next power of 2 for kernel efficiency.
        topk: Number of top blocks to select (must be power of 2).
        padding: Virtual padding ``(pad_head, pad_tail)`` so that
            ``pad_head + N + pad_tail`` divides ``score_block_size``; QKV data unchanged.

    Returns:
        o: Attention output, same rank as q ([B, N, C] or [B, H, N, D]).
        topk_indices: TopK block indices [B, H, N, topk].
        topk_scores: TopK block scores [B, H, N, topk].
    """
    input_4d = q.ndim == 4
    if input_4d:
        B, H, N, D = q.shape
        if num_heads is not None and num_heads != H:
            raise ValueError(f"num_heads ({num_heads}) != q.shape[1] ({H})")
        num_heads = H
        C = H * D
        if k.shape != (B, H, N, D) or v.shape != (B, H, N, D):
            raise ValueError("q, k, v must all be [B, H, N, D] for 4D input")
        q = q.transpose(1, 2).reshape(B, N, C).contiguous()
        k = k.transpose(1, 2).reshape(B, N, C).contiguous()
        v = v.transpose(1, 2).reshape(B, N, C).contiguous()
    else:
        if q.ndim != 3:
            raise ValueError("q must be 3D [B, N, C] or 4D [B, H, N, D]")
        if num_heads is None:
            raise ValueError("num_heads is required for 3D [B, N, C] input")

    o, topk_indices, topk_scores = FlashScoringFunction.apply(
        q, k, v, num_heads, score_block_size, topk, padding
    )
    if input_4d:
        B, N, C = o.shape
        H = num_heads
        D = C // H
        o = o.view(B, N, H, D).transpose(1, 2).contiguous()
    return o, topk_indices, topk_scores


def _flash_topk_score_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: Optional[int],
    block_size: int,
    topk: int,
    scale: Optional[float] = None,
    padding: Tuple[int, int] = (0, 0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash Attention with TopK block scoring (naive reference implementation).

    Args:
        q: Query [B, N, C] or [B, H, N, D].
        k, v: Same layout as q.
        num_heads: Required for 3D input; optional for 4D (inferred from shape[1]).
        block_size: Tokens per score block (``score_block_size``).
        topk: Number of top blocks to select.
        scale: Attention scale factor, defaults to 1/sqrt(D).
        padding: ``(pad_head, pad_tail)`` for virtual padding (same as ``flash_topk_score``).

    Returns:
        o: Attention output [B, H, N, D] (same layout as input if 4D).
        topk_indices: TopK block indices [B, H, N, topk].
        topk_scores: TopK block scores [B, H, N, topk].
    """
    dtype = q.dtype
    input_4d = q.ndim == 4
    if input_4d:
        B, H, N, D = q.shape
        if num_heads is not None and num_heads != H:
            raise ValueError(f"num_heads ({num_heads}) != q.shape[1] ({H})")
        num_heads = H
        C = H * D
        if k.shape != (B, H, N, D) or v.shape != (B, H, N, D):
            raise ValueError("q, k, v must all be [B, H, N, D] for 4D input")
        q = q.transpose(1, 2).reshape(B, N, C)
        k = k.transpose(1, 2).reshape(B, N, C)
        v = v.transpose(1, 2).reshape(B, N, C)
    else:
        B, N, C = q.shape
        if num_heads is None:
            raise ValueError("num_heads is required for 3D [B, N, C] input")

    H = num_heads
    D = C // H
    pad_head, pad_tail = padding
    if pad_head < 0 or pad_tail < 0:
        raise ValueError("padding must be non-negative (pad_head, pad_tail)")
    if pad_head >= block_size or pad_tail >= block_size:
        raise ValueError(
            f"padding ({pad_head}, {pad_tail}) must each be < block_size ({block_size})"
        )
    n_padded = pad_head + N + pad_tail

    assert k.shape == (B, N, C), f"k shape {k.shape} != {(B, N, C)}"
    assert v.shape == (B, N, C), f"v shape {v.shape} != {(B, N, C)}"
    assert C % H == 0, f"C={C} not divisible by num_heads={H}"
    if n_padded % block_size != 0:
        raise ValueError(
            f"pad_head + N + pad_tail = {n_padded} must be divisible by block_size ({block_size})"
        )
    M = n_padded // block_size
    assert topk <= M, f"topk={topk} must be <= M={M}"

    if scale is None:
        scale = D ** -0.5

    q = rearrange(q, 'b n (h d) -> b h n d', h=H)
    k = rearrange(k, 'b n (h d) -> b h n d', h=H)
    v = rearrange(v, 'b n (h d) -> b h n d', h=H)

    q, k, v = map(lambda x: x.float(), (q, k, v))

    scores = torch.einsum('bhqd,bhkd->bhqk', q * scale, k)
    attn_weights = scores.softmax(dim=-1)
    o = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

    if pad_head == 0 and pad_tail == 0:
        attn_blocks = rearrange(attn_weights, 'b h q (m bs) -> b h q m bs', bs=block_size)
        block_scores = attn_blocks.sum(dim=-1)
    else:
        block_scores = torch.zeros(B, H, N, M, device=q.device, dtype=q.dtype)
        for m in range(M):
            start = max(0, m * block_size - pad_head)
            end = min(N, (m + 1) * block_size - pad_head)
            if start < end:
                count = end - start
                block_scores[..., m] = attn_weights[..., start:end].sum(dim=-1) / float(count)

    topk_scores, topk_indices = torch.topk(
        block_scores, k=min(topk, M), dim=-1, largest=True, sorted=True
    )
    if input_4d:
        o = o.transpose(1, 2).contiguous()
    return o.to(dtype), topk_indices, topk_scores