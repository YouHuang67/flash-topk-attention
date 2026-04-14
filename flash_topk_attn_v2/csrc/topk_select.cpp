#include <torch/extension.h>

void topk_select_launch(
    torch::Tensor block_scores,
    torch::Tensor topk_indices,
    torch::Tensor topk_raw_scores,
    torch::Tensor topk_avg_scores,
    int M, int M_OUT,
    int score_block_size, int pad_head, int pad_tail,
    float threshold, int num_slices
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk_select_launch", &topk_select_launch);
}
