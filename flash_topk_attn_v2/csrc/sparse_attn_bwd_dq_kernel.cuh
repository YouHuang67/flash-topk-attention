#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>

#include "sparse_attn_kernel.cuh"

using namespace cute;


template <int kBlockM, int kBlockN, int kHeadDim, int kNWarps, typename Element>
struct SparseAttnBwdDqTypes {
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
    static constexpr int kBlockKGmem =
        (kBytePerRow % 128 == 0 ? 128
         : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kSwizzle =
        kBlockKGmem == 128 ? 4
        : (kBlockKGmem == 64 ? 3 : (kBlockKGmem == 32 ? 2 : 1));
    static constexpr int kSwizzleBase = sizeof(Element) == 2 ? 3 : 4;

    using SmemLayoutAtom = decltype(
        composition(Swizzle<kSwizzle, kSwizzleBase, kSwizzleBase>{},
                    Layout<Shape<_8, Int<kBlockKGmem>>,
                           Stride<Int<kBlockKGmem>, _1>>{}));

    using SmemLayoutQ = decltype(
        tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutdO = decltype(
        tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemLayoutK = decltype(
        tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutV = decltype(
        tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kHeadDim>>{}));
    using SmemLayoutVt = decltype(
        composition(SmemLayoutV{},
                    make_ordered_layout(
                        make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
                        Step<_2, _1>{})));
    using SmemLayoutKt = decltype(
        composition(SmemLayoutK{},
                    make_ordered_layout(
                        make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
                        Step<_2, _1>{})));
    using SmemLayoutdQ = decltype(
        tile_to_shape(SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, Element>;

    using GmemCopyAtom =
        Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, Element>;

    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    using GmemLayoutAtom = Layout<
        Shape<Int<NumMmaThreads / kGmemThreadsPerRow>,
              Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(GmemCopyAtom{}, GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    using SmemCopyAtomO = Copy_Atom<
        AutoVectorizingCopyWithAssumedAlignment<128>, Element>;

    static constexpr int kNRows = 2 * (2 * kBlockM / NumMmaThreads);

    struct SharedStorage {
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdQ>> smem_dq;
        };
        union {
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutdO>> smem_do;
            cute::array_aligned<Element, cute::cosize_v<SmemLayoutV>> smem_v;
        };
        cute::array_aligned<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    };
};
