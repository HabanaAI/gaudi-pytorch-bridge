#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <iostream>

using namespace std;

extern torch::Tensor embedding_bag_sum_hpu_wrap(
    const torch::Tensor& input,
    const torch::Tensor& indices,
    const torch::Tensor& offsets,
    const torch::Tensor& valid_count,
    int64_t kernel_mode);

extern torch::Tensor& embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
    torch::Tensor& out,
    const torch::Tensor& input,
    const torch::Tensor& indices_bwd,
    const torch::Tensor& offsets_bwd,
    const torch::Tensor& valid_count_bwd,
    int64_t kernel_mode);

// Input tensor 1	Input feature map	BF16/FP32	2D
// Input tensor 2	Indices Tensor	I32	1D
// Input tensor 3	Valid Count Tensor	I32	1D
// Enum kernel_mode
// Output tensor 1	Output feature map	FP32	2D
// GUID: gather_with_valid_count_2d_<bf16/ f32>
/*typedef enum {
  EMBEDDING_BAG_MODE_SUM = 0,
  EMBEDDING_BAG_MODE_SUM_SMALL_LENGTHS = 1
} HabanaEmbeddingBagKernelMode_t;
*/
torch::Tensor embedding_bag_sum_with_valid_count_f32(
    torch::Tensor input,
    torch::Tensor indices,
    torch::Tensor offsets,
    torch::Tensor validCount,
    int64_t kernelMode) {
  torch::Tensor out;

  out = embedding_bag_sum_hpu_wrap(
      input, indices, offsets, validCount, kernelMode);
  return out;
}

torch::Tensor& embedding_bag_sum_bwd_with_valid_count_f32(
    torch::Tensor& out,
    torch::Tensor& input,
    torch::Tensor& indices,
    torch::Tensor& offsets,
    torch::Tensor& validCount,
    int64_t kernelMode) {
  embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap(
      out, input, indices, offsets, validCount, kernelMode);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &embedding_bag_sum_with_valid_count_f32,
      "embedding bag sum forward");
  m.def(
      "backward",
      &embedding_bag_sum_bwd_with_valid_count_f32,
      "embedding bag sum bwd");
}
