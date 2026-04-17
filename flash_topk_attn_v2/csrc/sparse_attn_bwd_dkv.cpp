#include <torch/extension.h>

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
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_attn_bwd_dkv_launch", &sparse_attn_bwd_dkv_launch);
}
