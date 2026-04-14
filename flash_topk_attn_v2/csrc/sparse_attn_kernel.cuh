#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/cub.cuh>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

using namespace cute;


template<int THREADS>
struct SparseAllreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return SparseAllreduce<OFFSET>::run(x, op);
    }
};

template<>
struct SparseAllreduce<2> {
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};

struct SparseMaxOp {
    __device__ __forceinline__ float operator()(float const &x, float const &y) {
        return max(x, y);
    }
};

struct SparseSumOp {
    __device__ __forceinline__ float operator()(float const &x, float const &y) {
        return x + y;
    }
};


template<bool zero_init, typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void sparse_thread_reduce(
    Tensor<Engine0, Layout0> const &tensor,
    Tensor<Engine1, Layout1> &summary,
    Operator &op
) {
    static_assert(Layout0::rank == 2);
    static_assert(Layout1::rank == 1);
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor); ni++) {
        #pragma unroll
        for (int mi = 0; mi < size<0>(tensor); mi++) {
            summary(mi) = zero_init && ni == 0 ? tensor(mi, ni) : op(summary(mi), tensor(mi, ni));
        }
    }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ __forceinline__ void sparse_quad_allreduce(
    Tensor<Engine0, Layout0> &dst,
    Tensor<Engine1, Layout1> &src,
    Operator &op
) {
    #pragma unroll
    for (int i = 0; i < size(dst); i++) {
        dst(i) = SparseAllreduce<4>::run(src(i), op);
    }
}


template <bool Scale_max, bool Check_inf, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sparse_scale_apply_exp2(
    Tensor<Engine0, Layout0> &tensor,
    Tensor<Engine1, Layout1> const &max_val,
    float const scale
) {
    static_assert(Layout0::rank == 2);
    static_assert(Layout1::rank == 1);
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float const max_scaled = Check_inf
            ? (max_val(mi) == -INFINITY ? 0.f : max_val(mi) * scale)
            : max_val(mi) * scale;
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni) {
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
        }
    }
}


template <int kNRows>
struct SparseSoftmax {
    using TensorT = decltype(make_tensor<float>(Shape<Int<kNRows>>{}));
    TensorT row_max, row_sum;
    float const softmax_scale_log2;

    __device__ SparseSoftmax(float scale_log2) : softmax_scale_log2(scale_log2) {};

    template<bool Is_first, bool Check_inf, typename Tensor0>
    __forceinline__ __device__ TensorT max_get_scale(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol_sm80(acc_s.layout()));
        TensorT scores_scale;
        if constexpr (Is_first) {
            SparseMaxOp max_op;
            sparse_thread_reduce<true>(scores, row_max, max_op);
            sparse_quad_allreduce(row_max, row_max, max_op);
            cute::fill(scores_scale, 1.f);
        } else {
            TensorT scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            SparseMaxOp max_op;
            sparse_thread_reduce<false>(scores, row_max, max_op);
            sparse_quad_allreduce(row_max, row_max, max_op);
            #pragma unroll
            for (int mi = 0; mi < size(row_max); ++mi) {
                float scores_max_cur = !Check_inf
                    ? row_max(mi)
                    : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = exp2f((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);
                row_sum(mi) *= scores_scale(mi);
            }
        }
        return scores_scale;
    }

    template<bool Is_first, bool Check_inf, typename Tensor0>
    __forceinline__ __device__ void online_softmax(Tensor0 &acc_s) {
        Tensor scores = make_tensor(acc_s.data(), convert_layout_acc_rowcol_sm80(acc_s.layout()));
        sparse_scale_apply_exp2<true, Check_inf>(scores, row_max, softmax_scale_log2);
        SparseSumOp sum_op;
        sparse_thread_reduce<Is_first>(scores, row_sum, sum_op);
    }

    __forceinline__ __device__ TensorT finalize() {
        SparseSumOp sum_op;
        sparse_quad_allreduce(row_sum, row_sum, sum_op);
        TensorT scores_scale;
        #pragma unroll
        for (int mi = 0; mi < size(row_sum); ++mi) {
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1.f / sum;
            scores_scale(mi) = inv_sum;
            row_sum(mi) = (sum == 0.f || sum != sum) ? -INFINITY
                : row_max(mi) * (softmax_scale_log2 * float(M_LN2)) + __logf(sum);
        }
        return scores_scale;
    }

    template<typename Tensor1>
    __forceinline__ __device__ void rescale_o(Tensor1 &acc_o, TensorT const &scores_scale) {
        Tensor acc_o_rowcol = make_tensor(acc_o.data(), convert_layout_acc_rowcol_sm80(acc_o.layout()));
        #pragma unroll
        for (int mi = 0; mi < size<0>(acc_o_rowcol); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(acc_o_rowcol); ++ni) {
                acc_o_rowcol(mi, ni) *= scores_scale(mi);
            }
        }
    }

    static __device__ auto convert_layout_acc_rowcol_sm80(auto acc_layout) {
        static_assert(decltype(size<0>(acc_layout))::value == 4);
        static_assert(decltype(rank(acc_layout))::value == 3);
        auto l = logical_divide(acc_layout, Shape<_2>{});
        return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
    }
};


template <typename Element, typename EngineIn, typename LayoutIn, typename EngineOut>
__device__ __forceinline__ void sparse_convert_type_out(
    Tensor<EngineIn, LayoutIn> const &tensor,
    Tensor<EngineOut, LayoutIn> &out
) {
    using From_type = typename EngineIn::value_type;
    using To_type = typename EngineOut::value_type;
    static constexpr int FragmentSize = sizeof(From_type) >= sizeof(To_type)
        ? sizeof(From_type) / sizeof(To_type)
        : sizeof(To_type) / sizeof(From_type);
    static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0);
    Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
    Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
    cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
    #pragma unroll
    for (int i = 0; i < size(frag); ++i) { out_frg[i] = convert_op(frag[i]); }
}


template <bool A_in_regs,
          typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB, typename Hook>
__device__ __forceinline__ void sparse_gemm_sm80(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
    Tensor3 const& tCsA, Tensor4 const& tCsB,
    TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B,
    Hook hook
) {
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        if constexpr (!std::is_same_v<Hook, std::nullptr_t>) {
            if (i == 0) { hook(); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}


template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
__device__ __forceinline__ void sparse_gemm_rs_sm80(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB,
    Tensor3 const& tCsB,
    TiledMma tiled_mma,
    TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B
) {
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}


template <int N>
__device__ __forceinline__ void sparse_cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
struct SparseAttnTypes {
    static constexpr int NumMmaThreads = kNWarps * 32;

    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<Element, cutlass::bfloat16_t>,
        MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>,
        MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>
    >;

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>, _1, _1>>,
        Tile<Int<16 * kNWarps>, _16, _16>
    >;

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kBytePerRow = kHeadDim * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kSwizzle = kBlockKGmem == 128 ? 4 : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 2 ? 3 : 4;

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));

    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt = decltype(
        composition(SmemLayoutV{},
                    make_ordered_layout(
                        make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
                        Step<_2, _1>{})));
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

    using GmemCopyAtom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    using GmemLayoutAtom = Layout<
        Shape<Int<NumMmaThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(GmemCopyAtom{}, GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    using GmemTiledCopyO = decltype(
        make_tiled_copy(
            Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>{},
            GmemLayoutAtom{},
            Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    static constexpr int kMaxSortElems = 512;
    static constexpr int kSortBlockSize = kNWarps * 32;
    static constexpr int kSortIPT = kMaxSortElems / kSortBlockSize;

    struct SharedStorage {
        union {
            typename cub::BlockRadixSort<int32_t, kSortBlockSize, kSortIPT>::TempStorage sort;
            struct {
                int32_t sorted_indices[kMaxSortElems];
                union {
                    struct {
                        union {
                            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
                            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
                        };
                        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
                    } mainloop;
                    cute::array_aligned<Element, cute::cosize_v<SmemLayoutO>> smem_o;
                };
            };
        };
    };

    static constexpr int kNRows = 2 * (2 * kBlockM / NumMmaThreads);
};
