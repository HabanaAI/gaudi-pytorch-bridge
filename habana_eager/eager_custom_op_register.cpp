/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include "common/dump_args.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/optimizer_lamb_gen.h"

namespace habana {
namespace eager {
std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8(
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    bool,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::cast_to_fp8 is not available in Eager mode.");
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax) {
  TORCH_CHECK(false, "hpu::cast_to_fp8_v2 is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_cast_transpose(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_cast_transpose is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(
      false, "hpu::fp8_cast_transpose_bgrad is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad_dgelu(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const c10::optional<at::Tensor>&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(
      false,
      "hpu::fp8_cast_transpose_bgrad_dgelu is not available in Eager mode.");
}

at::Tensor cast_from_fp8(const at::Tensor&, const at::Tensor&, at::ScalarType) {
  TORCH_CHECK(false, "hpu::cast_from_fp8 is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_dropout(
    const at::Tensor&,
    double,
    const c10::optional<at::Tensor>&,
    bool,
    bool) {
  TORCH_CHECK(false, "hpu::fp8_dropout is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_gelu(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_gelu is not available in Eager mode.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_gelu_v2(
    const at::Tensor&,
    const at::Tensor&,
    bool,
    bool) {
  TORCH_CHECK(false, "hpu::fp8_gelu_v2 is not available in Eager mode.");
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> fp8_layernorm(
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    double,
    const at::Tensor&,
    bool,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_layernorm is not available in Eager mode.");
}

at::Tensor& fp8_gemm(
    const at::Tensor&,
    bool,
    const at::Tensor&,
    bool,
    const at::Tensor&,
    at::ScalarType,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    bool,
    at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_gemm is not available in Eager mode.");
}

at::Tensor fp8_gemm_v2(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate) {
  TORCH_CHECK(false, "hpu::fp8_gemm_v2 is not available in Eager mode.");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_bgrad_dgelu(
    const at::Tensor& grad,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    bool is_amax) {
  TORCH_CHECK(false, "hpu::fp8_bgrad_dgelu is not available in Eager mode.");
}

at::Tensor fp8_reshape(const at::Tensor& input, at::IntArrayRef shape) {
  TORCH_CHECK(false, "hpu::fp8_reshape is not available in Eager mode.");
}

at::Tensor& fp8_transpose(const at::Tensor&, at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_transpose is not available in Eager mode.");
}

at::Tensor& fp8_permute(
    const at::Tensor& input,
    at::IntArrayRef dims,
    at::Tensor& out) {
  TORCH_CHECK(false, "hpu::fp8_permute is not available in Eager mode.");
}

at::Tensor optimizer_lamb_fused_norm(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm) {
  PT_EAGER_TRACE;

  EagerOptimizerLambFusedNorm<at::Tensor> hpu_op{
      "hpu::optimizer_lamb_fused_norm", {grad, max_grad_norm}};
  return hpu_op.call();
}

void optimizer_resource_apply_momentum(
    at::TensorList params_momentum_buf_list,
    const at::TensorList dp_list,
    const double momentum) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_resource_apply_momentum :",
      DUMP_3ARGS(params_momentum_buf_list, dp_list, momentum));

  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_resource_apply_momentum",
      {params_momentum_buf_list, dp_list, momentum}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut,
       "hpu::optimizer_resource_apply_momentum",
       {0}});

  hpu_op.call(params_momentum_buf_list);
}

void optimizer_lars(
    const at::TensorList params,
    at::TensorList grads,
    c10::ArrayRef<int64_t> skip_masks,
    const double eeta,
    const double weight_decay,
    const double eps,
    const at::Tensor& lr) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_lars :",
      DUMP_7ARGS(params, grads, skip_masks, eeta, weight_decay, eps, lr));

  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_lars",
      {params, grads, skip_masks, eeta, weight_decay, eps, lr}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut, "hpu::optimizer_lars", {1}});

  hpu_op.call(grads);
}

void optimizer_lamb_fused_phase2(
    at::TensorList weights,
    const at::TensorList adam_norms,
    const at::TensorList weight_norms,
    const at::TensorList adam_steps,
    const double step,
    const double weight_decay,
    const bool use_lamb) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_lamb_fused_phase2:",
      DUMP_7ARGS(
          weights,
          adam_norms,
          weight_norms,
          adam_steps,
          step,
          weight_decay,
          use_lamb));

  EagerOp<void> hpu_op{
      "hpu::optimizer_lamb_fused_phase2",
      {weights,
       adam_norms,
       weight_norms,
       adam_steps,
       step,
       weight_decay,
       use_lamb}};
  return hpu_op.call(weights);
}

void optimizer_ema(
    const at::TensorList model_inputs,
    at::TensorList updated_ema,
    const at::Tensor& decay) {
  PT_EAGER_TRACE;
  PT_OP_INFO(" optimizer_ema :", DUMP_3ARGS(model_inputs, updated_ema, decay));

  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_ema", {model_inputs, updated_ema, decay}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut, "hpu::optimizer_ema", {1}});

  hpu_op.call(updated_ema);
}

void optimizer_sgd(
    const at::TensorList gradients,
    at::TensorList weights,
    at::Tensor& lr,
    const double wd,
    const double mom,
    const double damp,
    const bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd:",
      DUMP_7ARGS(gradients, weights, lr, wd, mom, damp, nesterov));
  TORCH_CHECK(
      (weights.size() > 0),
      "optimizer_sgd : can not process empty weight vector");
  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd", {gradients, weights, lr, wd, mom, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut, "hpu::optimizer_sgd", {1}});
  hpu_op.call({weights});
}

void optimizer_sgd_momentum(
    const at::TensorList gradients,
    at::TensorList weights,
    at::TensorList momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    at::Tensor& mom,
    const double wd,
    const double damp,
    const bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd_momentum:",
      DUMP_9ARGS(
          gradients,
          weights,
          momentum,
          epoch_num,
          lr,
          wd,
          mom,
          damp,
          nesterov));
  TORCH_CHECK(
      (weights.size() > 0),
      "optimizer_sgd_momentum : can not process empty weight vector");
  // auto mom_t = get_tensor_for_scalar(mom);
  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd_momentum",
      {gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut,
       "hpu::optimizer_sgd_momentum",
       {1, 2}});
  hpu_op.call({weights, momentum});
}

TORCH_LIBRARY(hpu, m) {
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::cast_to_fp8_v2(Tensor input, Tensor? scale, bool stochastic_rounding, bool is_amax) -> (Tensor, Tensor)");
  m.def(
      "hpu::fp8_cast_transpose(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) bgrad, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad_dgelu(Tensor grad, Tensor input, Tensor scale, Tensor? retain, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) bgrad, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor scale, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::fp8_dropout(Tensor input, float p, Tensor? scale, bool stochastic_rounding, bool is_amax) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_gelu(Tensor input, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) retain, Tensor(c!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_gelu_v2(Tensor input, Tensor scale, bool stochastic_rounding, bool is_amax) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_bgrad_dgelu(Tensor grad, Tensor input, Tensor? scale, Tensor? retain, bool stochastic_rounding, bool is_amax) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_layernorm(Tensor input, Tensor weight, Tensor bias, float eps, Tensor scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) mean, Tensor(c!) istd, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_gemm(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate) -> Tensor");
  m.def("hpu::fp8_reshape(Tensor input, int[] shape) -> Tensor");
  m.def("hpu::fp8_transpose(Tensor input, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::fp8_permute(Tensor input, int[] dims, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::optimizer_lamb_fused_norm(Tensor[] grad, float max_norm) -> Tensor");
  m.def(
      "hpu::optimizer_resource_apply_momentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
  m.def(
      "hpu::optimizer_lars(Tensor[] params, Tensor(a!)[] grads, int[] skip_masks, float eeta, float weight_decay, float eps, Tensor(b!) lr) -> ()");
  m.def(
      "hpu::optimizer_lamb_fused_phase2(Tensor(a!)[] weights, Tensor[] adam_norms, Tensor[] weight_norms, Tensor[] adam_steps, float step, float wd, bool use_lamb) -> ()");
  m.def(
      "hpu::optimizer_ema(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor(b!) decay) -> ()");
  m.def(
      "hpu::optimizer_sgd(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!) learning_rate, float wd, float mom, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::optimizer_sgd_momentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor mom, float wd, float damp, bool nesterov) -> ()");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::cast_to_fp8", cast_to_fp8);
  m.impl("hpu::cast_to_fp8_v2", cast_to_fp8_v2);
  m.impl("hpu::fp8_cast_transpose", fp8_cast_transpose);
  m.impl("hpu::fp8_cast_transpose_bgrad", fp8_cast_transpose_bgrad);
  m.impl("hpu::fp8_cast_transpose_bgrad_dgelu", fp8_cast_transpose_bgrad_dgelu);
  m.impl("hpu::cast_from_fp8", cast_from_fp8);
  m.impl("hpu::fp8_dropout", fp8_dropout);
  m.impl("hpu::fp8_gelu", fp8_gelu);
  m.impl("hpu::fp8_gelu_v2", fp8_gelu_v2);
  m.impl("hpu::fp8_bgrad_dgelu", fp8_bgrad_dgelu);
  m.impl("hpu::fp8_layernorm", fp8_layernorm);
  m.impl("hpu::fp8_gemm", fp8_gemm);
  m.impl("hpu::fp8_gemm_v2", fp8_gemm_v2);
  m.impl("hpu::fp8_reshape", fp8_reshape);
  m.impl("hpu::fp8_transpose", fp8_transpose);
  m.impl("hpu::fp8_permute", fp8_permute);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_fused_norm);
  m.impl(
      "hpu::optimizer_resource_apply_momentum",
      optimizer_resource_apply_momentum);
  m.impl("hpu::optimizer_lars", optimizer_lars);
  m.impl("hpu::optimizer_lamb_fused_phase2", optimizer_lamb_fused_phase2);
  m.impl("hpu::optimizer_ema", optimizer_ema);
  m.impl("hpu::optimizer_sgd", optimizer_sgd);
  m.impl("hpu::optimizer_sgd_momentum", optimizer_sgd_momentum);
}

} // namespace eager
} // namespace habana