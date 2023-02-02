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

// Wrappers to match signatures
static void optimizer_ResourceApplyMomentum(
    std::vector<at::Tensor>& params_momentum_buffer_vec,
    const std::vector<at::Tensor>& d_p_vec,
    const float momentum) {
  at::TensorList params_momentum_buffer_list(params_momentum_buffer_vec);
  at::TensorList d_p_list(d_p_vec);

  optimizer_ResourceApplyMomentum_hpu_wrap(
      params_momentum_buffer_list, d_p_list, momentum);
}

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

static void optimizer_fused_adamw(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    at::Tensor& neg_step,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay) {
  at::TensorList gradients(gradient_vec);
  at::TensorList weights(weight_vec);
  at::TensorList exp_avg(exp_avg_vec);
  at::TensorList exp_avg_sq(exp_avg_sq_vec);

  optimizer_adamw_hpu_wrap(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      lr,
      neg_step,
      beta1,
      beta2,
      epsilon,
      weight_decay);
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
    const float mom,
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

at::Tensor& habana_fp8_gemm_wrap_py(
    const at::Tensor& A,
    const at::Tensor& A_scale_inv,
    bool trans_A,
    const at::Tensor& B,
    const at::Tensor& B_scale_inv,
    bool trans_B,
    at::Tensor& out,
    py::object out_dtype,
    const at::Tensor& bias,
    bool accumulate) {
  return habana_fp8_gemm_wrap(
      A,
      A_scale_inv,
      trans_A,
      B,
      B_scale_inv,
      trans_B,
      out,
      torch::python::detail::py_object_to_dtype(out_dtype),
      bias,
      accumulate,
      out);
}
at::Tensor habana_cast_from_fp8_wrap_py(
    const at::Tensor& input,
    const at::Tensor& scale,
    py::object out_dtype) {
  return habana_cast_from_fp8_wrap(
      input, scale, torch::python::detail::py_object_to_dtype(out_dtype));
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

at::Tensor linear_ex_wrap_py(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    py::object dtype) {
  return linear_ex_wrap(
      input,
      weight,
      bias_opt,
      torch::python::detail::py_object_to_dtype(dtype));
}
std::vector<at::Tensor> linear_ex_backward_wrap_py(
    const at::Tensor& grad_output,
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias_opt,
    const c10::optional<at::Tensor>& bias_grad_opt,
    py::object dtype) {
  return linear_ex_backward_wrap(
      grad_output,
      input,
      weight,
      bias_opt,
      bias_grad_opt,
      torch::python::detail::py_object_to_dtype(dtype));
}

at::Tensor habana_random_seed(const at::Tensor& input) {
  return habana_random_seed_wrap(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  //////////////////////////// Optimizers /////////////////////////////////////
  m.def(
      "fused_adamw",
      &optimizer_fused_adamw,
      "Compute and apply gradient update to parameters for Adam optimizer");
  m.def(
      "fused_lamb_phase1",
      &optimizer_lamb_phase1_hpu_wrap,
      "Compute and apply gradient update to parameters for lamb optimizer phase1");
  m.def(
      "fused_lamb_phase2",
      &optimizer_lamb_phase2_hpu_wrap,
      "Compute and apply gradient update to parameters for lamb optimizer phase2");
  m.def(
      "fused_lamb_norm",
      &optimizer_lamb_fused_norm_hpu_wrap,
      "Compute and apply global grad norm for lamb optimizer");
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
      "Optimizer Sparse Adagrad with valid count ");
  m.def("fused_lars", &optimizer_fused_lars, "Optimizer Fused Lars");
  m.def(
      "fused_resource_apply_momentum",
      &optimizer_ResourceApplyMomentum,
      "Optimizer Fused Resource Apply Momentum");
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
#if IS_PYTORCH_FORK_AT_LEAST(1, 0)
  m.def(
      "cast_to_fp8",
      &habana_cast_to_fp8_wrap,
      "Cast FP32 or BF16 to lower precission with optional stochastic rounding");
#endif
  m.def(
      "cast_to_fp8_te",
      &habana_cast_to_fp8_te_wrap,
      "Cast FP32 or BF16 to FP8 with scaling input and optional stochastic rounding");
  m.def(
      "cast_from_fp8",
      &habana_cast_from_fp8_wrap_py,
      "Cast FP8 to FP32 or BF16 with scaling input");
  m.def("fp8_gemm", &habana_fp8_gemm_wrap_py, "fp8 GEMM with inputs scaling");
  m.def(
      "fp8_transpose", &habana_fp8_transpose_wrap, "Transposes 2D fp8 tensor");
  m.def("matmul_ex", &matmul_ex_wrap_py, "Matmul with explicit dtype");
  m.def(
      "matmul_ex_backward",
      &matmul_ex_backward_wrap_py,
      "MatmulBackward with explicit dtype");
  m.def("linear_ex", &linear_ex_wrap_py, "Linear with explicit dtype");
  m.def(
      "linear_ex_backward",
      &linear_ex_backward_wrap_py,
      "LinearBackward with explicit dtype");
  m.def(
      "random_seed",
      &habana_random_seed,
      "Sets random seed in the LFSR register");
}
