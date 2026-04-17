#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cub/cub.cuh>


__global__ void iota_kernel(int32_t* out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) out[idx] = idx;
}


void sort_by_count_launch(
    torch::Tensor reverse_counts,
    torch::Tensor sorted_counts,
    torch::Tensor sorted_global_ids,
    int total,
    int end_bit
) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto global_ids = torch::empty(
        {total}, torch::TensorOptions().dtype(torch::kInt32)
                                       .device(reverse_counts.device()));

    {
        constexpr int kThreads = 256;
        int blocks = (total + kThreads - 1) / kThreads;
        iota_kernel<<<blocks, kThreads, 0, stream>>>(
            global_ids.data_ptr<int32_t>(), total);
    }

    const int32_t* keys_in = reverse_counts.data_ptr<int32_t>();
    int32_t* keys_out = sorted_counts.data_ptr<int32_t>();
    const int32_t* values_in = global_ids.data_ptr<int32_t>();
    int32_t* values_out = sorted_global_ids.data_ptr<int32_t>();

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, temp_bytes,
        keys_in, keys_out, values_in, values_out, total,
        0, end_bit, stream);

    auto temp = torch::empty(
        {(int64_t)temp_bytes},
        torch::TensorOptions().dtype(torch::kUInt8)
                              .device(reverse_counts.device()));

    cub::DeviceRadixSort::SortPairsDescending(
        temp.data_ptr(), temp_bytes,
        keys_in, keys_out, values_in, values_out, total,
        0, end_bit, stream);
}
