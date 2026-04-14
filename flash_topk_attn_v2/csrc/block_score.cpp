#include <torch/extension.h>

void block_score_launch(
    torch::Tensor q, torch::Tensor k,
    torch::Tensor m_locals, torch::Tensor l_locals,
    torch::Tensor block_scores,
    int B, int N, int N_phys, int H, int D,
    int M, int score_block_size,
    float softmax_scale,
    int pad_head
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_score_launch", &block_score_launch);
}
