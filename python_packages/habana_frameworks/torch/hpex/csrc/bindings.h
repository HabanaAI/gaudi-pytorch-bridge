#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
embedding_bag_preproc(
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    int64_t embeddingTableLen);
