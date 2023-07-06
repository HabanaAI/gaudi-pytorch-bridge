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
#include "hpu_ops/masked_batch_gemm.h"
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

std::tuple<at::Tensor, at::Tensor> fp8_fast_softmax(
    const at::Tensor& input,
    const at::Tensor& mask,
    const c10::optional<at::Tensor>& scale,
    double softmax_scale,
    bool stochastic_rounding,
    bool is_amax) {
  TORCH_CHECK(false, "hpu::fp8_fast_softmax is not available in Eager mode.");
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

at::Tensor optimizer_lamb_norm(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm) {
  PT_EAGER_TRACE;

  EagerOptimizerLambNorm<at::Tensor> hpu_op{
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

void optimizer_lamb_phase1(
    const at::TensorList gradients,
    const at::TensorList weights,
    at::TensorList exp_avg,
    at::TensorList exp_avg_sq,
    at::TensorList out_weight_norms,
    at::TensorList out_adam_norms,
    at::TensorList out_adam_steps,
    const at::Tensor clip_global_grad_norm,
    const int64_t grad_averaging,
    const double beta1,
    const double beta2,
    const double epsilon,
    const at::Tensor bias_correction1,
    const at::Tensor bias_correction2,
    const double weight_decay) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_lamb_phase1:",
      DUMP_12ARGS(
          gradients,
          weights,
          exp_avg,
          exp_avg_sq,
          clip_global_grad_norm,
          grad_averaging,
          beta1,
          beta2,
          epsilon,
          bias_correction1,
          bias_correction2,
          weight_decay));

  EagerOp<void> hpu_op{
      "hpu::optimizer_lamb_phase1",
      {gradients,
       weights,
       exp_avg,
       exp_avg_sq,
       out_weight_norms,
       out_adam_norms,
       out_adam_steps,
       clip_global_grad_norm,
       grad_averaging,
       beta1,
       beta2,
       epsilon,
       bias_correction1,
       bias_correction2,
       weight_decay}};
  return hpu_op.call(
      {exp_avg, exp_avg_sq, out_weight_norms, out_adam_norms, out_adam_steps});
}

void optimizer_lamb_phase2(
    at::TensorList weights,
    const at::TensorList adam_norms,
    const at::TensorList weight_norms,
    const at::TensorList adam_steps,
    const at::Tensor& neg_step,
    const double weight_decay,
    const bool use_lamb) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_lamb_phase2:",
      DUMP_7ARGS(
          weights,
          adam_norms,
          weight_norms,
          adam_steps,
          neg_step,
          weight_decay,
          use_lamb));

  EagerOp<void> hpu_op{
      "hpu::optimizer_lamb_phase2",
      {weights,
       adam_norms,
       weight_norms,
       adam_steps,
       neg_step,
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

void optimizer_adamw(
    const at::TensorList gradient_vec,
    at::TensorList weight_vec,
    at::TensorList exp_avg_vec,
    at::TensorList exp_avg_sq_vec,
    const at::Tensor& neg_step_t,
    const double beta1,
    const double beta2,
    const double epsilon,
    const at::Tensor& weight_decay,
    const bool has_weight_decay) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      DUMP_10ARGS(
          gradient_vec,
          weight_vec,
          exp_avg_vec,
          exp_avg_sq_vec,
          neg_step_t,
          beta1,
          beta2,
          epsilon,
          weight_decay,
          has_weight_decay));

  TORCH_CHECK(
      (weight_vec.size() > 0),
      "optimizer_adamw : can not process empty weight vector");

  eager::EagerOp<void> hpu_op{
      "hpu::optimizer_adamw",
      {gradient_vec,
       weight_vec,
       exp_avg_vec,
       exp_avg_sq_vec,
       neg_step_t,
       beta1,
       beta2,
       epsilon,
       weight_decay,
       has_weight_decay}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::InplaceOut,
       "hpu::optimizer_adamw",
       {1, 2, 3}});

  hpu_op.call({weight_vec, exp_avg_vec, exp_avg_sq_vec});
}

at::Tensor rotary_pos_embedding(
    const at::Tensor& input,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const int64_t offset,
    const int64_t mode) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rotary_pos_embedding :", DUMP_5ARGS(input, sin, cos, offset, mode));

  eager::EagerOp<at::Tensor> hpu_op{
      "hpu::rotary_pos_embedding",
      {input, sin, cos, offset, mode},
      {input.sizes().vec()},
      0};

  return hpu_op.call();
}

at::Tensor rotary_pos_embedding_backward(
    const at::Tensor& grad_in,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const int64_t offset) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rotary_pos_embedding_backward :", DUMP_4ARGS(grad_in, sin, cos, offset));

  eager::EagerOp<at::Tensor> hpu_op{
      "hpu::rotary_pos_embedding_backward",
      {grad_in, sin, cos, offset},
      {grad_in.sizes().vec()},
      0};

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> rms_norm(
    const at::Tensor& data_in,
    const at::Tensor& gamma,
    double epsilon) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO("rms_norm :", DUMP_3ARGS(data_in, gamma, epsilon));

  std::vector<int64_t> inverse_root_mean_square_sizes{data_in.sizes().vec()};
  inverse_root_mean_square_sizes.back() = 1;

  eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::rms_norm",
      {data_in, gamma, epsilon},
      {data_in.sizes().vec(), inverse_root_mean_square_sizes},
      0};
  hpu_op.set_scalar_types({data_in.scalar_type(), c10::ScalarType::Float});

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> rms_norm_backward(
    const at::Tensor& grad_in,
    const at::Tensor& data_in,
    const at::Tensor& gamma,
    const at::Tensor& inverse_rms) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rms_norm_backward :", DUMP_4ARGS(grad_in, data_in, gamma, inverse_rms));

  eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::rms_norm_backward",
      {grad_in, data_in, gamma, inverse_rms},
      {data_in.sizes().vec(), gamma.sizes().vec()},
      0};

  return hpu_op.call();
}

at::Tensor masked_batch_gemm(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& mask_a,
    const at::Tensor& mask_b,
    bool trans_a,
    bool trans_b) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "masked_batch_gemm :",
      DUMP_6ARGS(a, b, mask_a, mask_b, trans_a, trans_b));

  eager::EagerOp<at::Tensor> hpu_op{
      "hpu::masked_batch_gemm",
      {a, b, mask_a, mask_b, trans_a, trans_b},
      MaskedBatchGemmOutputShape};

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> retain_softmax_producer(
    const at::Tensor& self) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO("retain_softmax_producer :", DUMP_ARG(self));

  auto out_shape = self.sizes().vec();
  auto retain_output_shape = out_shape;
  retain_output_shape.back() = 1;
  eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::retain_softmax_producer",
      {self},
      {out_shape, retain_output_shape, retain_output_shape},
      0};
  hpu_op.set_scalar_types(
      {self.scalar_type(), self.scalar_type(), c10::ScalarType::Float});

  return hpu_op.call();
}

at::Tensor retain_softmax_consumer(
    const at::Tensor& self,
    const at::Tensor& max,
    const at::Tensor& exp_sum_recpr) {
  PT_OP_TRACE;
  PT_EAGER_TRACE;
  PT_OP_INFO("retain_softmax_consumer :", DUMP_3ARGS(self, max, exp_sum_recpr));

  eager::EagerOp<at::Tensor> hpu_op{
      "hpu::retain_softmax_consumer",
      {self, max, exp_sum_recpr},
      {self.sizes().vec()},
      0};

  return hpu_op.call();
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
      "hpu::fp8_fast_softmax(Tensor input, Tensor mask, Tensor? scale, float softmax_scale, bool stochastic_rounding, bool is_amax) -> (Tensor, Tensor)");
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
      "hpu::optimizer_lars(Tensor[] params, Tensor(a!)[] grads, int[] skip_masks, float eeta, float weight_decay, float eps, Tensor lr) -> ()");
  m.def(
      "hpu::optimizer_lamb_phase1(Tensor[] gradients, Tensor[] weights, Tensor(a!)[] exp_avg, Tensor(b!)[] exp_avg_sq, Tensor(c!)[] out_weight_norms, Tensor(d!)[] out_adam_norms, Tensor(e!)[] out_adam_steps, Tensor clip_global_grad_norm, int grad_averaging, float beta1, float beta2, float epsilon, Tensor bias_correction1, Tensor bias_correction2, float weight_decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_phase2(Tensor(a!)[] weights, Tensor[] adam_norms, Tensor[] weight_norms, Tensor[] adam_steps, Tensor neg_step, float wd, bool use_lamb) -> ()");
  m.def(
      "hpu::optimizer_ema(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "hpu::optimizer_adamw(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, Tensor weight_decay, bool has_weight_decay) -> ()");
  m.def(
      "hpu::rotary_pos_embedding(Tensor input, Tensor sin, Tensor cos, int offset, int mode) -> Tensor");
  m.def(
      "hpu::rotary_pos_embedding_backward(Tensor grad_in, Tensor sin, Tensor cos, int offset) -> Tensor");
  m.def(
      "hpu::rms_norm(Tensor data_in, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
  m.def(
      "hpu::rms_norm_backward(Tensor grad_in, Tensor data_in, Tensor gamma, Tensor inverse_rms) -> (Tensor, Tensor)");
  m.def(
      "hpu::masked_batch_gemm(Tensor a, Tensor b, Tensor mask_a, Tensor mask_b, bool trans_a, bool trans_b) -> Tensor");
  m.def("control_edge_(Tensor(a) self)-> Tensor(a)");
  m.def(
      "hpu::retain_softmax_producer(Tensor self) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::retain_softmax_consumer(Tensor self, Tensor max, Tensor exp_sum_recpr) -> Tensor");
  m.def("hpu::view(Tensor input, Tensor shape) -> Tensor");
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::topk(Tensor self, Tensor k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor(a!) values, Tensor(b!) indices)");
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
  m.impl("hpu::fp8_fast_softmax", fp8_fast_softmax);
  m.impl("hpu::fp8_layernorm", fp8_layernorm);
  m.impl("hpu::fp8_gemm", fp8_gemm);
  m.impl("hpu::fp8_gemm_v2", fp8_gemm_v2);
  m.impl("hpu::fp8_reshape", fp8_reshape);
  m.impl("hpu::fp8_transpose", fp8_transpose);
  m.impl("hpu::fp8_permute", fp8_permute);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_norm);
  m.impl(
      "hpu::optimizer_resource_apply_momentum",
      optimizer_resource_apply_momentum);
  m.impl("hpu::optimizer_lars", optimizer_lars);
  m.impl("hpu::optimizer_lamb_phase1", optimizer_lamb_phase1);
  m.impl("hpu::optimizer_lamb_phase2", optimizer_lamb_phase2);
  m.impl("hpu::optimizer_ema", optimizer_ema);
  m.impl("hpu::optimizer_adamw", optimizer_adamw);
  m.impl("hpu::rotary_pos_embedding", rotary_pos_embedding);
  m.impl("hpu::rotary_pos_embedding_backward", rotary_pos_embedding_backward);
  m.impl("hpu::rms_norm", rms_norm);
  m.impl("hpu::rms_norm_backward", rms_norm_backward);
  m.impl("hpu::masked_batch_gemm", masked_batch_gemm);
  m.impl("hpu::retain_softmax_producer", retain_softmax_producer);
  m.impl("hpu::retain_softmax_consumer", retain_softmax_consumer);
}

} // namespace eager
} // namespace habana
