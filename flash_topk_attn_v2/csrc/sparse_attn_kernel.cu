#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "sparse_attn_kernel.cuh"


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
__global__ void __launch_bounds__(kNWarps * 32, 2)
sparse_attn_fwd_kernel(
    const Element* __restrict__ q_ptr,
    const Element* __restrict__ k_ptr,
    const Element* __restrict__ v_ptr,
    const int32_t* __restrict__ merged_indices_ptr,
    const int32_t* __restrict__ counts_ptr,
    Element* __restrict__ o_ptr,
    float* __restrict__ softmax_max_ptr,
    float* __restrict__ softmax_lse_ptr,
    int B, int N, int N_phys, int H, int D,
    int QM, int qblock_topk,
    int q_block_size, int kv_block_size,
    int stride_q_b, int stride_q_n, int stride_q_h,
    int stride_k_b, int stride_k_n, int stride_k_h,
    int stride_v_b, int stride_v_n, int stride_v_h,
    int stride_o_b, int stride_o_n, int stride_o_h,
    float softmax_scale_log2,
    int q_pad_head, int kv_pad_head
) {
    using Types = SparseAttnTypes<kBlockM, kBlockN, kHeadDim, kNWarps, Element>;
    using SharedStorage = typename Types::SharedStorage;

    extern __shared__ char smem_buf[];
    SharedStorage& smem = *reinterpret_cast<SharedStorage*>(smem_buf);

    int const qm_idx = blockIdx.x;
    int const bh_idx = blockIdx.z;
    int const b_idx = bh_idx / H;
    int const h_idx = bh_idx % H;
    int const thread_idx = threadIdx.x;

    int const count = __ldg(counts_ptr + b_idx * H * QM + h_idx * QM + qm_idx);
    if (count <= 0) {
        int const q_global_start = qm_idx * q_block_size - q_pad_head;
        for (int row = thread_idx; row < q_block_size; row += kNWarps * 32) {
            int const global_row = q_global_start + row;
            if (global_row >= 0 && global_row < N) {
                softmax_max_ptr[b_idx * H * N + h_idx * N + global_row] = -INFINITY;
                softmax_lse_ptr[b_idx * H * N + h_idx * N + global_row] = -INFINITY;
                for (int d = 0; d < kHeadDim; d++) {
                    o_ptr[b_idx * stride_o_b + global_row * stride_o_n
                          + h_idx * stride_o_h + d] = Element(0);
                }
            }
        }
        return;
    }

    static constexpr int kSortIPT = Types::kSortIPT;
    static constexpr int kMaxSortElems = Types::kMaxSortElems;

    int64_t indices_base = (int64_t)b_idx * H * QM * qblock_topk
                         + (int64_t)h_idx * QM * qblock_topk
                         + (int64_t)qm_idx * qblock_topk;
    const int32_t* block_indices_ptr;

    if (qblock_topk <= kMaxSortElems) {
        int32_t sort_keys[kSortIPT];
        #pragma unroll
        for (int ipt = 0; ipt < kSortIPT; ++ipt) {
            int elem_idx = thread_idx * kSortIPT + ipt;
            int32_t val = INT_MAX;
            if (elem_idx < qblock_topk) {
                val = __ldg(merged_indices_ptr + indices_base + elem_idx);
                if (val < 0) val = INT_MAX;
            }
            sort_keys[ipt] = val;
        }

        using BlockSort = cub::BlockRadixSort<int32_t, kNWarps * 32, kSortIPT>;
        BlockSort(smem.sort).Sort(sort_keys);
        __syncthreads();

        #pragma unroll
        for (int ipt = 0; ipt < kSortIPT; ++ipt) {
            int elem_idx = thread_idx * kSortIPT + ipt;
            int32_t val = sort_keys[ipt];
            if (elem_idx < count && val >= 0 && val != INT_MAX) {
                smem.sorted_indices[elem_idx] = val;
            }
        }
        __syncthreads();
        block_indices_ptr = smem.sorted_indices;
    } else {
        block_indices_ptr = merged_indices_ptr + indices_base;
    }

    using TiledMma = typename Types::TiledMma;
    using SmemLayoutQ = typename Types::SmemLayoutQ;
    using SmemLayoutK = typename Types::SmemLayoutK;
    using SmemLayoutV = typename Types::SmemLayoutV;
    using SmemLayoutVt = typename Types::SmemLayoutVt;
    using SmemLayoutO = typename Types::SmemLayoutO;
    using SmemCopyAtom = typename Types::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Types::SmemCopyAtomTransposed;
    using GmemTiledCopyQKV = typename Types::GmemTiledCopyQKV;
    using GmemTiledCopyO = typename Types::GmemTiledCopyO;
    using SmemCopyAtomO = typename Types::SmemCopyAtomO;
    static constexpr int NumMmaThreads = Types::NumMmaThreads;
    static constexpr int kNRows = Types::kNRows;

    Tensor sQ = make_tensor(make_smem_ptr(smem.mainloop.smem_q.data()), SmemLayoutQ{});
    Tensor sK = make_tensor(make_smem_ptr(smem.mainloop.smem_k.data()), SmemLayoutK{});
    Tensor sV = make_tensor(make_smem_ptr(smem.mainloop.smem_v.data()), SmemLayoutV{});
    Tensor sVt = make_tensor(make_smem_ptr(smem.mainloop.smem_v.data()), SmemLayoutVt{});

    int const q_start = qm_idx * q_block_size - q_pad_head;
    int q_load_start = q_start;
    if (q_load_start < 0) q_load_start = 0;
    if (q_load_start + kBlockM > N_phys) q_load_start = N_phys - kBlockM;
    int const q_row_offset = q_start - q_load_start;
    int const q_offset = b_idx * stride_q_b + q_load_start * stride_q_n + h_idx * stride_q_h;
    Tensor mQ = make_tensor(
        make_gmem_ptr(q_ptr + q_offset),
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_stride(stride_q_n, _1{}));

    GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(thread_idx);

    {
        Tensor tQgQ = gmem_thr_copy_QKV.partition_S(mQ);
        Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
        cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
    }
    cute::cp_async_fence();

    auto load_K = [&](int idx, int sub) {
        int n_block = block_indices_ptr[idx];
        int token_start = n_block * kv_block_size + sub * kBlockN - kv_pad_head;
        if (token_start < 0) token_start = 0;
        if (token_start + kBlockN > N_phys) token_start = N_phys - kBlockN;
        int k_offset = b_idx * stride_k_b + token_start * stride_k_n + h_idx * stride_k_h;
        Tensor gK = make_tensor(
            make_gmem_ptr(k_ptr + k_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_k_n, _1{}));
        Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
        Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
        cute::copy(gmem_tiled_copy_QKV, tKgK, tKsK);
        cute::cp_async_fence();
    };

    auto load_V = [&](int idx, int sub) {
        int n_block = block_indices_ptr[idx];
        int token_start = n_block * kv_block_size + sub * kBlockN - kv_pad_head;
        if (token_start < 0) token_start = 0;
        if (token_start + kBlockN > N_phys) token_start = N_phys - kBlockN;
        int v_offset = b_idx * stride_v_b + token_start * stride_v_n + h_idx * stride_v_h;
        Tensor gV = make_tensor(
            make_gmem_ptr(v_ptr + v_offset),
            Shape<Int<kBlockN>, Int<kHeadDim>>{},
            make_stride(stride_v_n, _1{}));
        Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
        Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
        cute::copy(gmem_tiled_copy_QKV, tVgV, tVsV);
        cute::cp_async_fence();
    };

    load_K(0, 0);

    sparse_cp_async_wait<0>();
    __syncthreads();

    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_slice(thread_idx);

    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

    auto smem_tiled_copy_Q = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(thread_idx);
    auto smem_tiled_copy_K = make_tiled_copy_B(SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(thread_idx);
    auto smem_tiled_copy_V = make_tiled_copy_B(SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(thread_idx);

    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    {
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        Tensor tSsQ_copy_view = smem_thr_copy_Q.partition_S(sQ);
        cute::copy(smem_tiled_copy_Q, tSsQ_copy_view, tSrQ_copy_view);
    }

    Tensor tOrO = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(tOrO);

    SparseSoftmax<kNRows> softmax(softmax_scale_log2);

    int const kv_tiles = (kv_block_size + kBlockN - 1) / kBlockN;
    int const kv_tail = kv_block_size - (kv_tiles - 1) * kBlockN;
    int const total_iters = count * kv_tiles;

    auto taccScO = thr_mma.partition_C(cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{}));
    auto taccScO_rowcol = make_tensor(taccScO.data(),
        SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccScO.layout()));

    for (int iter = 0; iter < total_iters; ++iter) {
        int const idx = iter / kv_tiles;
        int const sub = iter % kv_tiles;

        int const n_block = block_indices_ptr[idx];
        int const intended_start = n_block * kv_block_size + sub * kBlockN - kv_pad_head;
        int actual_start = intended_start;
        if (actual_start < 0) actual_start = 0;
        if (actual_start + kBlockN > N_phys) actual_start = N_phys - kBlockN;
        int const col_offset = intended_start - actual_start;
        int const valid_cols = min(kBlockN, kv_block_size - sub * kBlockN);

        Tensor tSrS = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(tSrS);

        sparse_cp_async_wait<0>();
        __syncthreads();

        auto load_V_hook = [&]() {
            load_V(idx, sub);
        };

        Tensor tSrQ_cur = tSrQ;
        Tensor tSrK = thr_mma.partition_fragment_B(sK);
        sparse_gemm_sm80<true>(
            tSrS, tSrQ_cur, tSrK, tSsQ, tSsK,
            tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K, load_V_hook);

        __syncthreads();
        if (iter + 1 < total_iters) {
            int const next_idx = (iter + 1) / kv_tiles;
            int const next_sub = (iter + 1) % kv_tiles;
            load_K(next_idx, next_sub);
        }

        bool need_col_mask = (col_offset > 0 || valid_cols < kBlockN
                              || intended_start < 0
                              || intended_start + valid_cols > N);
        if (need_col_mask) {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                #pragma unroll
                for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                    int col = get<1>(taccScO_rowcol(mi, ni));
                    int kv_global = intended_start + (col - col_offset);
                    if (col < col_offset || col >= col_offset + valid_cols
                        || kv_global < 0 || kv_global >= N) {
                        tSrS_rowcol(mi, ni) = -INFINITY;
                    }
                }
            }
        }

        if (q_block_size < kBlockM || q_pad_head > 0) {
            Tensor tSrS_rowcol = make_tensor(tSrS.data(),
                SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(tSrS.layout()));
            #pragma unroll
            for (int mi = 0; mi < size<0>(tSrS_rowcol); ++mi) {
                int row = get<0>(taccScO_rowcol(mi, _0{}));
                int global_row = q_start + (row - q_row_offset);
                if (row < q_row_offset
                    || row >= q_row_offset + q_block_size
                    || global_row < 0
                    || global_row >= N) {
                    #pragma unroll
                    for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                        tSrS_rowcol(mi, ni) = -INFINITY;
                    }
                }
            }
        }

        if (iter == 0) {
            Tensor scores_scale = softmax.template max_get_scale<true, true>(tSrS);
            softmax.template online_softmax<true, true>(tSrS);
        } else {
            Tensor scores_scale = softmax.template max_get_scale<false, false>(tSrS);
            softmax.template online_softmax<false, false>(tSrS);
            softmax.rescale_o(tOrO, scores_scale);
        }

        Tensor tOrP_acc = make_tensor(tSrS.data(),
            SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(tSrS.layout()));

        using LayoutAcc = decltype(tSrS.layout());
        auto convert_layout_acc_Aregs_sm80 = [](LayoutAcc acc_layout) {
            auto l = logical_divide(acc_layout, Shape<Underscore, Underscore, _2>{});
            return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
        };

        Tensor tOrP_Aregs = make_tensor(tSrS.data(), convert_layout_acc_Aregs_sm80(tSrS.layout()));
        Tensor tOrP = make_tensor_like<Element>(tOrP_Aregs);
        sparse_convert_type_out<Element>(tOrP_Aregs, tOrP);

        sparse_cp_async_wait<0>();
        __syncthreads();

        Tensor tOrV = thr_mma.partition_fragment_B(sVt);
        sparse_gemm_rs_sm80(tOrO, tOrP, tOrV, tOsVt,
                            tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    Tensor scores_scale = softmax.finalize();
    softmax.rescale_o(tOrO, scores_scale);

    __syncthreads();

    Tensor sO = make_tensor(make_smem_ptr(smem.smem_o.data()), SmemLayoutO{});

    {
        auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
        auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(thread_idx);

        Tensor tOrO_out = make_tensor_like<Element>(tOrO);
        sparse_convert_type_out<Element>(tOrO, tOrO_out);

        Tensor taccOrO = smem_thr_copy_O.retile_S(tOrO_out);
        Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
        cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    }
    __syncthreads();

    {
        for (int row = thread_idx; row < q_block_size; row += kNWarps * 32) {
            int const global_row = q_start + row;
            if (global_row >= 0 && global_row < N) {
                int const o_base = b_idx * stride_o_b + global_row * stride_o_n + h_idx * stride_o_h;
                for (int d = 0; d < kHeadDim; d++) {
                    o_ptr[o_base + d] = sO(q_row_offset + row, d);
                }
            }
        }
    }

    {
        auto taccOcO = thr_mma.partition_C(cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{}));
        auto taccOcO_rowcol = make_tensor(taccOcO.data(),
            SparseSoftmax<kNRows>::convert_layout_acc_rowcol_sm80(taccOcO.layout()));
        auto taccOcO_row = taccOcO_rowcol(_, _0{});

        #pragma unroll
        for (int mi = 0; mi < size(taccOcO_row); ++mi) {
            int const tile_row = get<0>(taccOcO_row(mi));
            int const global_row = q_start + (tile_row - q_row_offset);
            if (get<1>(taccOcO_row(_0{})) == 0
                && tile_row >= q_row_offset
                && tile_row < q_row_offset + q_block_size
                && global_row >= 0
                && global_row < N) {
                int const lse_offset = b_idx * H * N + h_idx * N + global_row;
                softmax_max_ptr[lse_offset] = softmax.softmax_max_val(mi);
                softmax_lse_ptr[lse_offset] = softmax.softmax_lse_val(mi);
            }
        }
    }
}


void sparse_attn_fwd_launch(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor merged_indices, torch::Tensor counts,
    torch::Tensor o, torch::Tensor softmax_max, torch::Tensor softmax_lse,
    int B, int N, int N_phys, int H, int D,
    int QM, int qblock_topk,
    int q_block_size, int kv_block_size,
    float softmax_scale,
    int q_pad_head, int kv_pad_head
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    int kNWarps;
    if (q_block_size <= 16)       kNWarps = 1;
    else if (q_block_size <= 32)  kNWarps = 2;
    else                          kNWarps = 4;

    dim3 grid(QM, 1, B * H);
    dim3 block(kNWarps * 32);

    float softmax_scale_log2 = softmax_scale * float(M_LOG2E);

    int stride_q_b = q.stride(0);
    int stride_q_n = q.stride(1);
    int stride_k_b = k.stride(0);
    int stride_k_n = k.stride(1);
    int stride_v_b = v.stride(0);
    int stride_v_n = v.stride(1);
    int stride_o_b = o.stride(0);
    int stride_o_n = o.stride(1);

    bool is_bf16 = (q.scalar_type() == at::ScalarType::BFloat16);

    #define LAUNCH(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS, Elem)                    \
        do {                                                                     \
            using SharedStorage = typename SparseAttnTypes<                       \
                BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS, Elem>::SharedStorage;        \
            int smem_size = sizeof(SharedStorage);                               \
            auto kernel = sparse_attn_fwd_kernel<                                \
                BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS, Elem>;                      \
            if (smem_size > 48 * 1024) {                                         \
                cudaFuncSetAttribute(                                             \
                    kernel,                                                       \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);      \
            }                                                                     \
            kernel<<<grid, block, smem_size, stream>>>(                           \
                reinterpret_cast<const Elem*>(q.data_ptr()),                      \
                reinterpret_cast<const Elem*>(k.data_ptr()),                      \
                reinterpret_cast<const Elem*>(v.data_ptr()),                      \
                merged_indices.data_ptr<int32_t>(),                               \
                counts.data_ptr<int32_t>(),                                       \
                reinterpret_cast<Elem*>(o.data_ptr()),                            \
                softmax_max.data_ptr<float>(),                                   \
                softmax_lse.data_ptr<float>(),                                   \
                B, N, N_phys, H, D, QM, qblock_topk,                              \
                q_block_size, kv_block_size,                                      \
                stride_q_b, stride_q_n, D,                                        \
                stride_k_b, stride_k_n, D,                                        \
                stride_v_b, stride_v_n, D,                                        \
                stride_o_b, stride_o_n, D,                                        \
                softmax_scale_log2,                                               \
                q_pad_head, kv_pad_head);                                         \
        } while (0)

    #define DISPATCH_DTYPE(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS)                   \
        do {                                                                     \
            if (is_bf16) { LAUNCH(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS,           \
                                  cutlass::bfloat16_t); }                        \
            else         { LAUNCH(BLOCK_M, BLOCK_N, HEAD_DIM, NWARPS,           \
                                  cutlass::half_t); }                            \
        } while (0)

    #define DISPATCH_NWARPS(HEAD_DIM)                                            \
        do {                                                                     \
            if (kNWarps == 1)      DISPATCH_DTYPE(16,  64, HEAD_DIM, 1);         \
            else if (kNWarps == 2) DISPATCH_DTYPE(32,  64, HEAD_DIM, 2);         \
            else                   DISPATCH_DTYPE(64,  64, HEAD_DIM, 4);         \
        } while (0)

    if (D == 128)          DISPATCH_NWARPS(128);
    else if (D == 64)      DISPATCH_NWARPS(64);
    else if (D == 256)     DISPATCH_NWARPS(256);
    else if (D == 32)      DISPATCH_NWARPS(32);
    else if (D == 96)      DISPATCH_NWARPS(96);
    else if (D == 160)     DISPATCH_NWARPS(160);

    #undef LAUNCH
    #undef DISPATCH_DTYPE
    #undef DISPATCH_NWARPS
}
