/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <torch/extension.h>

#include "bindings.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
#include "transformer_engine/common.h"

// Wrappers to match signatures
static void optimizer_fused_lars(
    const std::vector<at::Tensor>& paramsVec,
    std::vector<at::Tensor>& gradsVec,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr) {
  at::TensorList params(paramsVec);
  at::TensorList grads(gradsVec);

  optimizer_lars_hpu_wrap(
      params, grads, skipMasks, eeta, weight_decay, eps, lr);
}

static void optimizer_fused_adagrad(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& variance_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);
  at::TensorList variances(variance_vec);

  optimizer_adagrad_hpu_wrap(
      gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);
}

static void optimizer_fused_ema(
    const std::vector<at::Tensor>& model_inputs,
    std::vector<at::Tensor>& updated_ema,
    const at::Tensor& decay) {
  at::TensorList modelInputs(model_inputs);
  at::TensorList updatedEma(updated_ema);
  optimizer_ema_hpu_wrap(modelInputs, updatedEma, decay);
}

static void optimizer_fused_sgd(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);

  optimizer_sgd_hpu_wrap(gradients, weights, lr, wd, mom, damp, nesterov);
}

static void optimizer_fused_sgd_momentum(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& momentum_vec,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    at::Tensor& mom,
    const float damp,
    const bool nesterov) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);
  at::TensorList momentum(momentum_vec);

  optimizer_sgd_momentum_hpu_wrap(
      gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);
}

static std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count(
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

at::Tensor matmul_ex_wrap_py(
    const at::Tensor& self,
    const at::Tensor& other,
    py::object dtype) {
  return matmul_ex_wrap(
      self, other, torch::python::detail::py_object_to_dtype(dtype));
}
std::tuple<at::Tensor, at::Tensor> matmul_ex_backward_wrap_py(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& other,
    py::object dtype) {
  return matmul_ex_backward_wrap(
      grad_output,
      self,
      other,
      torch::python::detail::py_object_to_dtype(dtype));
}

at::Tensor habana_random_seed(const at::Tensor& input) {
  return habana_random_seed_wrap(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //////////////////////////// Optimizers /////////////////////////////////////
  m.def(
      "fused_adagrad",
      &optimizer_fused_adagrad,
      "Compute and apply gradient update to parameters for Adagrad optimizer");
  m.def(
      "fused_sgd",
      &optimizer_fused_sgd,
      "Compute and apply gradient update to parameters for SGD optimizer");
  m.def(
      "fused_ema",
      &optimizer_fused_ema,
      "Compute and apply exponential moving avg update in ema optimizer");
  m.def(
      "fused_sgd_momentum",
      &optimizer_fused_sgd_momentum,
      "Compute and apply gradient update to parameters for SGD with momentum optimizer");
  m.def(
      "sparse_sgd_with_valid_count",
      &optimizer_sparse_sgd_with_valid_count,
      "Optimizer Sparse Stochastic Gradient Descent with valid count");
  m.def(
      "sparse_adagrad_with_valid_count",
      &optimizer_sparse_adagrad_with_valid_count_hpu_wrap,
      "Optimizer Sparse Adagrad with valid count");
  m.def("fused_lars", &optimizer_fused_lars, "Optimizer Fused Lars");
  //////////////////////////// Normalizations /////////////////////////////////
  m.def(
      "fused_norm",
      &fused_norm_hpu_wrap,
      "Compute the norm of the norm of the input vector of tensors");

  //////////////////////////// Kernels ////////////////////////////////////////
  m.def(
      "custom_nms",
      &torchvision_nms_hpu_wrap,
      "NMS operation for boxes of a single class");
  m.def(
      "batched_nms",
      &batched_nms_hpu_wrap,
      "NMS operation for boxes of a multiple classes");
  m.def(
      "embedding_bag_sum_fwd",
      &embedding_bag_sum_hpu_wrap,
      "embedding bag sum forward");
  m.def(
      "embedding_bag_sum_bwd",
      &embedding_bag_sum_bwd_out_kernel_mode_hpu_wrap,
      "embedding bag sum bwd");
  m.def(
      "embedding_bag_preproc", &embedding_bag_preproc, "embedding bag preproc");
  m.def(
      "roi_align_forward",
      &vision::ops::roi_align_fwd_wrap,
      "ROI Align forward");
  m.def(
      "roi_align_backward",
      &vision::ops::roi_align_bwd_wrap,
      "ROI Align backward");
  m.def("matmul_ex", &matmul_ex_wrap_py, "Matmul with explicit dtype");
  m.def(
      "matmul_ex_backward",
      &matmul_ex_backward_wrap_py,
      "MatmulBackward with explicit dtype");
  m.def(
      "random_seed",
      &habana_random_seed,
      "Sets random seed in the LFSR register");
  m.def(
      "permute_1D_sparse_data",
      &habana_permute_1D_sparse_data_wrap,
      "Permute 1D sparse data");
  m.def(
      "permute_2D_sparse_data",
      &habana_permute_2D_sparse_data_wrap,
      "Permute 2D sparse data");
  m.def(
      "expand_into_jagged_permute",
      &habana_expand_into_jagged_permute_wrap,
      "Expands the sparse data permute index from table dimension to batch dimension");
  m.def(
      "bounds_check_indices",
      &habana_bounds_check_indices_wrap,
      "Out of bounds checks");
  m.def(
      "split_permute_cat",
      &habana_split_permute_cat_wrap,
      "Replaces the combination of split_with_sizes and cat operators");

  // TE Data structures
  py::class_<transformer_engine::FP8TensorMeta>(m, "FP8TensorMeta")
      .def(py::init<>())
      .def_readwrite("scale", &transformer_engine::FP8TensorMeta::scale)
      .def_readwrite("scale_inv", &transformer_engine::FP8TensorMeta::scale_inv)
      .def_readwrite(
          "amax_history", &transformer_engine::FP8TensorMeta::amax_history)
      .def_readwrite(
          "amax_history_index",
          &transformer_engine::FP8TensorMeta::amax_history_index);

  py::enum_<transformer_engine::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", transformer_engine::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", transformer_engine::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM2_INPUT", transformer_engine::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", transformer_engine::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM3_INPUT", transformer_engine::FP8FwdTensors::GEMM3_INPUT)
      .value("GEMM3_WEIGHT", transformer_engine::FP8FwdTensors::GEMM3_WEIGHT)
      .value("GEMM4_INPUT", transformer_engine::FP8FwdTensors::GEMM4_INPUT)
      .value("GEMM4_WEIGHT", transformer_engine::FP8FwdTensors::GEMM4_WEIGHT)
      .value("GEMM5_INPUT", transformer_engine::FP8FwdTensors::GEMM5_INPUT)
      .value("GEMM5_WEIGHT", transformer_engine::FP8FwdTensors::GEMM5_WEIGHT);

  py::enum_<transformer_engine::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", transformer_engine::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_OUTPUT2", transformer_engine::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_OUTPUT3", transformer_engine::FP8BwdTensors::GRAD_OUTPUT3)
      .value("GRAD_OUTPUT4", transformer_engine::FP8BwdTensors::GRAD_OUTPUT4)
      .value("GRAD_OUTPUT5", transformer_engine::FP8BwdTensors::GRAD_OUTPUT5);
}
