#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>

#include <iostream>

using namespace std;

// Input tensors
// 1	Gradient             FP32/FP16/BF16	2D
// 2	Weights              FP32	2D
// 3	Moments              FP32	2D
// 4	Indices              I32	1D
// 5	Learning rate	       FP32	1D
// 6	Valid count	         I32	1D
//
// Output tensors
// 1	Weights              FP32	2D
// 2	Moments              FP32	2D

extern std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_wrap(
    const torch::Tensor& gradients,
    torch::Tensor& weights_in,
    torch::Tensor& moments_in,
    const torch::Tensor& indices,
    const torch::Tensor& learning_rate,
    const torch::Tensor& valid_count,
    float mom,
    bool nesterov);

std::tuple<torch::Tensor&, torch::Tensor&> optimizer_sparse_sgd_with_valid_count(
    const torch::Tensor& gradients,
    torch::Tensor& weights_in,
    torch::Tensor& moments_in,
    const torch::Tensor& indices,
    const torch::Tensor& learning_rate,
    const torch::Tensor& valid_count) {
  return optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count,
      0.0f,
      false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &optimizer_sparse_sgd_with_valid_count,
      "Optimizer Sparse Stochastic Gradient Descent with valid count ");
  m.def("backward", &optimizer_sparse_sgd_with_valid_count, "TO BE REMOVED");
}
