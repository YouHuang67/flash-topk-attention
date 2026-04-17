#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "sparse_attn_bwd_dkv_kernel.cuh"


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
__global__ void __launch_bounds__(kNWarps * 32, 2)
sparse_attn_bwd_dkv_sorted_kernel(
    const Element* __restrict__ q_ptr,
    const Element* __restrict__ k_ptr,
    const Element* __restrict__ v_ptr,
    const Element* __restrict__ do_ptr,
    const int32_t* __restrict__ reverse_indices_ptr,
    const int32_t* __restrict__ reverse_counts_ptr,
    const int32_t* __restrict__ sorted_kv_indices_ptr,
    const float* __restrict__ softmax_max_ptr,
    const float* __restrict__ softmax_lse_ptr,
    const float* __restrict__ delta_ptr,
    Element* __restrict__ dk_ptr,
    Element* __restrict__ dv_ptr,
    int B, int N, int N_phys, int H, int D,
    int QM, int M,
    int q_block_size, int kv_block_size,
    int stride_q_b, int stride_q_n, int stride_q_h,
    int stride_k_b, int stride_k_n, int stride_k_h,
    int stride_v_b, int stride_v_n, int stride_v_h,
    int stride_do_b, int stride_do_n, int stride_do_h,
    int stride_dk_b, int stride_dk_n, int stride_dk_h,
    int stride_dv_b, int stride_dv_n, int stride_dv_h,
    float softmax_scale_log2,
    float softmax_scale,
    int q_pad_head, int kv_pad_head
) {
    using Types = SparseAttnBwdDkvTypes<kBlockM, kBlockN, kHeadDim, kNWarps, Element>;
    using SharedStorage = typename Types::SharedStorage;

    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const bh_idx = blockIdx.z;
    int const b_idx = bh_idx / H;
    int const h_idx = bh_idx % H;
    int const m_idx = sorted_kv_indices_ptr[bh_idx * M + blockIdx.x];
    int const thread_idx = threadIdx.x;

    int const reverse_count = __ldg(reverse_counts_ptr + b_idx * H * M + h_idx * M + m_idx);
    if (reverse_count <= 0) {
        int const kv_global_start = m_idx * kv_block_size - kv_pad_head;
        for (int row = thread_idx; row < kv_block_size; row += kNWarps * 32) {
            int const global_row = kv_global_start + row;
            if (global_row >= 0 && global_row < N) {
                for (int d = 0; d < kHeadDim; d++) {
                    dk_ptr[b_idx * stride_dk_b + global_row * stride_dk_n
                           + h_idx * stride_dk_h + d] = Element(0);
                    dv_ptr[b_idx * stride_dv_b + global_row * stride_dv_n
                           + h_idx * stride_dv_h + d] = Element(0);
                }
            }
        }
        return;
    }

    int64_t reverse_base = (int64_t)b_idx * H * M * QM
                         + (int64_t)h_idx * M * QM
                         + (int64_t)m_idx * QM;
    const int32_t* qm_indices_ptr = reverse_indices_ptr + reverse_base;

    using TiledMma = typename Types::TiledMma;
    using SmemLayoutK = typename Types::SmemLayoutK;
    using SmemLayoutV = typename Types::SmemLayoutV;
    using SmemLayoutQ = typename Types::SmemLayoutQ;
    using SmemLayoutdO = typename Types::SmemLayoutdO;
    using SmemLayoutQt = typename Types::SmemLayoutQt;
    using SmemLayoutdOt = typename Types::SmemLayoutdOt;
    using SmemLayoutdK = typename Types::SmemLayoutdK;
    using SmemLayoutdV = typename Types::SmemLayoutdV;
    using SmemCopyAtom = typename Types::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Types::SmemCopyAtomTransposed;
    using GmemTiledCopyQKV = typename Types::GmemTiledCopyQKV;
    using SmemCopyAtomdKdV = typename Types::SmemCopyAtomdKdV;
    static constexpr int kNRows = Types::kNRows;

    Tensor sK = make_tensor(make_smem_ptr(smem.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(smem.smem_v.data()), SmemLayoutV{});
    Tensor sQ = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutQ{});
    Tensor sdO = make_tensor(make_smem_ptr(smem.smem_do.data()), SmemLayoutdO{});
    Tensor sQt = make_tensor(make_smem_ptr(smem.smem_q.data()), SmemLayoutQt{});
    Tensor sdOt = make_tensor(make_smem_ptr(smem.smem_do.data()), SmemLayoutdOt{});

    int const kv_start = m_idx * kv_block_size - kv_pad_head;
    int kv_load_start = kv_start;
    if (kv_load_start < 0) kv_load_start = 0;
    if (kv_load_start + kBlockM > N_phys) kv_load_start = N_phys - kBlockM;
    int const kv_row_offset = kv_start - kv_load_start;

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);

    {
        int const k_offset = b_idx * stride_k_b + kv_load_start * stride_k_n
                           + h_idx * stride_k_h;
        Tensor mK = make_tensor(
            make_gmem_ptr(k_ptr + k_offset),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_stride(stride_k_n, _1{}));
        Tensor tKgK = gmem_thr_copy_QKV.partition_S(mK);
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
    }
    cute::cp_async_fence();

    {
        int const v_offset = b_idx * stride_v_b + kv_load_start * stride_v_n
                           + h_idx * stride_v_h;
        Tensor mV = make_tensor(
            make_gmem_ptr(v_ptr + v_offset),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_stride(stride_v_n, _1{}));
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(mV);
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
    }
    cute::cp_async_fence();

    sparse_cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_A = smem_tiled_copy_A.get_thread_slice(thread_idx);
    auto smem_tiled_copy_B = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_B = smem_tiled_copy_B.get_thread_slice(thread_idx);
    auto smem_tiled_copy_Bt = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_Bt = smem_tiled_copy_Bt.get_thread_slice(thread_idx);

    Tensor tSsK = smem_thr_copy_A.partition_S(sK);
    Tensor tSsV = smem_thr_copy_A.partition_S(sV);
    Tensor tSsQ = smem_thr_copy_B.partition_S(sQ);
    Tensor tSsdO = smem_thr_copy_B.partition_S(sdO);
    Tensor tOsQt = smem_thr_copy_Bt.partition_S(sQt);
    Tensor tOsdOt = smem_thr_copy_Bt.partition_S(sdOt);

    auto taccScO = thr_mma.partition_C(
        cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto taccScO_rowcol = make_tensor(taccScO.data(),
        SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccScO.layout()));

    Tensor tOrdK = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(tOrdK);
    Tensor tOrdV = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(tOrdV);

    int const q_tiles = (q_block_size + kBlockN - 1) / kBlockN;

    auto load_Q = [&](int idx, int sub) {
        int qm_block = qm_indices_ptr[idx];
        int token_start = qm_block * q_block_size + sub * kBlockN - q_pad_head;
        if (token_start < 0) token_start = 0;
        if (token_start + kBlockN > N_phys) token_start = N_phys - kBlockN;
        int q_offset = b_idx * stride_q_b + token_start * stride_q_n
                     + h_idx * stride_q_h;
        Tensor gQ = make_tensor(
            make_gmem_ptr(q_ptr + q_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_q_n, _1{}));
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
        cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
        cute::cp_async_fence();
    };

    auto load_dO = [&](int idx, int sub) {
        int qm_block = qm_indices_ptr[idx];
        int token_start = qm_block * q_block_size + sub * kBlockN - q_pad_head;
        if (token_start < 0) token_start = 0;
        if (token_start + kBlockN > N_phys) token_start = N_phys - kBlockN;
        int do_offset = b_idx * stride_do_b + token_start * stride_do_n
                      + h_idx * stride_do_h;
        Tensor gdO = make_tensor(
            make_gmem_ptr(do_ptr + do_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_do_n, _1{}));
        Tensor tDOgDO = gmem_thr_copy_QKV.partition_S(gdO);
        Tensor tDOsDO = gmem_thr_copy_QKV.partition_D(sdO);
        cute::copy(gmem_tiled_copy_QKV, tDOgDO, tDOsDO);
        cute::cp_async_fence();
    };

    load_Q(0, 0);
    load_dO(0, 0);

    for (int iter = 0; iter < reverse_count * q_tiles; ++iter) {
        int const idx = iter / q_tiles;
        int const sub = iter % q_tiles;

        int const qm_block = qm_indices_ptr[idx];
        int const intended_start = qm_block * q_block_size
                                 + sub * kBlockN - q_pad_head;
        int actual_start = intended_start;
        if (actual_start < 0) actual_start = 0;
        if (actual_start + kBlockN > N_phys) actual_start = N_phys - kBlockN;
        int const col_offset = intended_start - actual_start;
        int const valid_cols = min(kBlockN, q_block_size - sub * kBlockN);

        sparse_cp_async_wait<0>();
        __syncthreads();

        Tensor tSrS = partition_fragment_C(
            tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);

        {
            Tensor tSrK = thr_mma.partition_fragment_A(sK);
            Tensor tSrQ = thr_mma.partition_fragment_B(sQ);
            sparse_gemm_sm80<false>(
                tSrS, tSrK, tSrQ, tSsK, tSsQ,
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                smem_thr_copy_A, smem_thr_copy_B, nullptr);
        }

        if (col_offset > 0 || valid_cols < kBlockN
            || intended_start < 0
            || intended_start + valid_cols > N) {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                    int col = get<1>(taccScO_rowcol(mi, ni));
                    int q_global = intended_start + (col - col_offset);
                    if (col < col_offset || col >= col_offset + valid_cols
                        || q_global < 0 || q_global >= N) {
                        tSrS_rowcol(mi, ni) = -INFINITY;
                    }
                }
            }
        }

        if (kv_block_size < kBlockM || kv_pad_head > 0 || kv_row_offset > 0) {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                int row = get<0>(taccScO_rowcol(mi, _0{}));
                int global_row = kv_start + (row - kv_row_offset);
                if (row < kv_row_offset
                    || row >= kv_row_offset + kv_block_size
                    || global_row < 0
                    || global_row >= N) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                        tSrS_rowcol(mi, ni) = -INFINITY;
                    }
                }
            }
        }

        {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));

            int const q_token_start = qm_block * q_block_size + sub * kBlockN - q_pad_head;
            #pragma unroll
            for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                int col = get<1>(taccScO_rowcol(_0{}, ni));
                int const tile_col = col - col_offset;
                int const q_idx = q_token_start + tile_col;

                float max_val = -INFINITY;
                float lse_val = 0.f;
                if (tile_col >= 0 && tile_col < valid_cols
                    && q_idx >= 0 && q_idx < N) {
                    max_val = softmax_max_ptr[b_idx * H * N + h_idx * N + q_idx];
                    lse_val = softmax_lse_ptr[b_idx * H * N + h_idx * N + q_idx];
                }

                #pragma unroll
                for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                    float s_scaled = tSrS_rowcol(mi, ni) * softmax_scale;
                    tSrS_rowcol(mi, ni) = (s_scaled == -INFINITY)
                        ? 0.f : expf(s_scaled - max_val - lse_val);
                }
            }
        }

        Tensor tSrdP = partition_fragment_C(
            tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrdP);

        {
            Tensor tSrV = thr_mma.partition_fragment_A(sV);
            Tensor tSrdO = thr_mma.partition_fragment_B(sdO);
            sparse_gemm_sm80<false>(
                tSrdP, tSrV, tSrdO, tSsV, tSsdO,
                tiled_mma, smem_tiled_copy_A, smem_tiled_copy_B,
                smem_thr_copy_A, smem_thr_copy_B, nullptr);
        }

        using LayoutAcc = decltype(tSrS.layout());
        auto convert_layout_acc_Aregs_sm80 = [](LayoutAcc acc_layout) {
            auto l = logical_divide(
                acc_layout, Shape<Underscore, Underscore, _2>{});
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)),
                               get<1>(l), get<2, 1>(l));
        };

        {
            Tensor tPregs = make_tensor(tSrS.data(),
                convert_layout_acc_Aregs_sm80(tSrS.layout()));
            Tensor tPelem = make_tensor_like<Element>(tPregs);
            sparse_convert_type_out<Element>(tPregs, tPelem);

            Tensor tOrdOt = thr_mma.partition_fragment_B(sdOt);
            sparse_gemm_rs_sm80(tOrdV, tPelem, tOrdOt, tOsdOt,
                                tiled_mma, smem_tiled_copy_Bt, smem_thr_copy_Bt);
        }

        {
            Tensor tP_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrS.layout()));
            Tensor tdP_rowcol = make_tensor(tSrdP.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                    tSrdP.layout()));

            int const q_token_start = qm_block * q_block_size + sub * kBlockN - q_pad_head;
            #pragma unroll
            for (int ni = 0; ni < size<1>(tP_rowcol); ++ni) {
                int col = get<1>(taccScO_rowcol(_0{}, ni));
                int const tile_col = col - col_offset;
                int const q_idx = q_token_start + tile_col;

                float delta_val = 0.f;
                if (tile_col >= 0 && tile_col < valid_cols && q_idx >= 0 && q_idx < N) {
                    delta_val = delta_ptr[b_idx * H * N + h_idx * N + q_idx];
                }

                #pragma unroll
                for (int mi = 0; mi < size<0>(tP_rowcol); ++mi) {
                    float p_val = tP_rowcol(mi, ni);
                    tP_rowcol(mi, ni) = p_val * (tdP_rowcol(mi, ni) - delta_val);
                }
            }
        }

        {
            Tensor tDSregs = make_tensor(tSrS.data(),
                convert_layout_acc_Aregs_sm80(tSrS.layout()));
            Tensor tDSelem = make_tensor_like<Element>(tDSregs);
            sparse_convert_type_out<Element>(tDSregs, tDSelem);

            Tensor tOrQt = thr_mma.partition_fragment_B(sQt);
            sparse_gemm_rs_sm80(tOrdK, tDSelem, tOrQt, tOsQt,
                                tiled_mma, smem_tiled_copy_Bt, smem_thr_copy_Bt);
        }

        __syncthreads();

        if (iter + 1 < reverse_count * q_tiles) {
            int const next_idx = (iter + 1) / q_tiles;
            int const next_sub = (iter + 1) % q_tiles;
            load_Q(next_idx, next_sub);
            load_dO(next_idx, next_sub);
        }
    }

    {
        Tensor dK_rowcol = make_tensor(tOrdK.data(),
            SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(
                tOrdK.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(dK_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dK_rowcol); ++ni) {
                dK_rowcol(mi, ni) *= softmax_scale;
            }
        }
    }

    __syncthreads();

    Tensor sdK = make_tensor(
        make_smem_ptr(smem.smem_dk.data()), SmemLayoutdK{});
    Tensor sdV = make_tensor(
        make_smem_ptr(smem.smem_dv.data()), SmemLayoutdV{});

    {
        auto smem_tiled_copy_dK = make_tiled_copy_C(SmemCopyAtomdKdV{}, tiled_mma);
        auto smem_thr_copy_dK = smem_tiled_copy_dK.get_thread_slice(thread_idx);

        Tensor tOrdK_out = make_tensor_like<Element>(tOrdK);
        sparse_convert_type_out<Element>(tOrdK, tOrdK_out);

        Tensor taccOrdK = smem_thr_copy_dK.retile_S(tOrdK_out);
        Tensor taccOsdK = smem_thr_copy_dK.partition_D(sdK);
        cute::copy(smem_tiled_copy_dK, taccOrdK, taccOsdK);
    }

    {
        auto smem_tiled_copy_dV = make_tiled_copy_C(SmemCopyAtomdKdV{}, tiled_mma);
        auto smem_thr_copy_dV = smem_tiled_copy_dV.get_thread_slice(thread_idx);

        Tensor tOrdV_out = make_tensor_like<Element>(tOrdV);
        sparse_convert_type_out<Element>(tOrdV, tOrdV_out);

        Tensor taccOrdV = smem_thr_copy_dV.retile_S(tOrdV_out);
        Tensor taccOsdV = smem_thr_copy_dV.partition_D(sdV);
        cute::copy(smem_tiled_copy_dV, taccOrdV, taccOsdV);
    }
    __syncthreads();

    {
        for (int row = thread_idx; row < kv_block_size; row += kNWarps * 32) {
            int const global_row = kv_start + row;
            if (global_row >= 0 && global_row < N) {
                int const dk_base = b_idx * stride_dk_b
                                  + global_row * stride_dk_n
                                  + h_idx * stride_dk_h;
                int const dv_base = b_idx * stride_dv_b
                                  + global_row * stride_dv_n
                                  + h_idx * stride_dv_h;
                for (int d = 0; d < kHeadDim; d++) {
                    dk_ptr[dk_base + d] = sdK(kv_row_offset + row, d);
                    dv_ptr[dv_base + d] = sdV(kv_row_offset + row, d);
                }
            }
        }
    }
}


void sparse_attn_bwd_dkv_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor do_tensor,
    torch::Tensor reverse_indices, torch::Tensor reverse_counts,
    torch::Tensor sorted_kv_indices,
    torch::Tensor softmax_max, torch::Tensor softmax_lse,
    torch::Tensor delta,
    torch::Tensor dk, torch::Tensor dv,
    int B, int N, int N_phys, int H, int D,
    int QM, int M,
    int q_block_size, int kv_block_size,
    float softmax_scale,
    int q_pad_head, int kv_pad_head
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int kNWarps = 4;
    constexpr int kBlockM = 64;
    constexpr int kBlockN = 32;

    dim3 grid(M, 1, B * H);
    dim3 block(kNWarps * 32);

    float softmax_scale_log2 = softmax_scale * float(M_LOG2E);

    int stride_q_b = q.stride(0);
    int stride_q_n = q.stride(1);
    int stride_k_b = k.stride(0);
    int stride_k_n = k.stride(1);
    int stride_v_b = v.stride(0);
    int stride_v_n = v.stride(1);
    int stride_do_b = do_tensor.stride(0);
    int stride_do_n = do_tensor.stride(1);
    int stride_dk_b = dk.stride(0);
    int stride_dk_n = dk.stride(1);
    int stride_dv_b = dv.stride(0);
    int stride_dv_n = dv.stride(1);

    bool is_bf16 = (q.scalar_type() == at::ScalarType::BFloat16);

    #define LAUNCH_DKV(HEAD_DIM, Elem)                                    \
        do {                                                                  \
            using SharedStorage = typename SparseAttnBwdDkvTypes<          \
                kBlockM, kBlockN, HEAD_DIM, kNWarps, Elem>::SharedStorage;   \
            int smem_size = sizeof(SharedStorage);                            \
            auto kernel = sparse_attn_bwd_dkv_sorted_kernel<                    \
                kBlockM, kBlockN, HEAD_DIM, kNWarps, Elem>;                 \
            cudaFuncSetAttribute(                                              \
                kernel,                                                        \
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);      \
            kernel<<<grid, block, smem_size, stream>>>(                       \
                reinterpret_cast<const Elem*>(q.data_ptr()),                  \
                reinterpret_cast<const Elem*>(k.data_ptr()),                  \
                reinterpret_cast<const Elem*>(v.data_ptr()),                  \
                reinterpret_cast<const Elem*>(do_tensor.data_ptr()),          \
                reverse_indices.data_ptr<int32_t>(),                          \
                reverse_counts.data_ptr<int32_t>(),                           \
                sorted_kv_indices.data_ptr<int32_t>(),                        \
                softmax_max.data_ptr<float>(),                                \
                softmax_lse.data_ptr<float>(),                                \
                delta.data_ptr<float>(),                                      \
                reinterpret_cast<Elem*>(dk.data_ptr()),                       \
                reinterpret_cast<Elem*>(dv.data_ptr()),                       \
                B, N, N_phys, H, D, QM, M,                                   \
                q_block_size, kv_block_size,                                  \
                stride_q_b, stride_q_n, D,                                    \
                stride_k_b, stride_k_n, D,                                    \
                stride_v_b, stride_v_n, D,                                    \
                stride_do_b, stride_do_n, D,                                  \
                stride_dk_b, stride_dk_n, D,                                  \
                stride_dv_b, stride_dv_n, D,                                  \
                softmax_scale_log2, softmax_scale,                            \
                q_pad_head, kv_pad_head);                                     \
        } while (0)

    #define DISPATCH_DKV_DTYPE(HEAD_DIM)                                  \
        do {                                                                  \
            if (is_bf16) { LAUNCH_DKV(HEAD_DIM, cutlass::bfloat16_t); }  \
            else         { LAUNCH_DKV(HEAD_DIM, cutlass::half_t); }      \
        } while (0)

    if (D == 128)          DISPATCH_DKV_DTYPE(128);
    else if (D == 64)      DISPATCH_DKV_DTYPE(64);
    else if (D == 256)     DISPATCH_DKV_DTYPE(256);
    else if (D == 32)      DISPATCH_DKV_DTYPE(32);
    else if (D == 48)      DISPATCH_DKV_DTYPE(48);
    else if (D == 80)      DISPATCH_DKV_DTYPE(80);
    else if (D == 96)      DISPATCH_DKV_DTYPE(96);
    else if (D == 160)     DISPATCH_DKV_DTYPE(160);

    #undef LAUNCH_DKV
    #undef DISPATCH_DKV_DTYPE
}
