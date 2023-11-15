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
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include "common/dump_args.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/masked_batch_gemm.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/optimizer_lamb_gen.h"

namespace {
using habana::to_string; // For DUMP_*ARGS

std::tuple<at::Tensor&, at::Tensor&> cast_to_fp8(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& amax) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "cast_to_fp8 :",
      DUMP_5ARGS(input, scale, stochastic_rounding, out, amax));

  TORCH_CHECK(
      out.scalar_type() != at::ScalarType::Char,
      "hpu::cast_to_fp8 with torch.int8 dtype is not available in Eager mode.");

  habana::eager::EagerOp<std::tuple<at::Tensor&, at::Tensor&>> hpu_op{
      "hpu::cast_to_fp8",
      {input, scale, stochastic_rounding, out, amax},
      {input.sizes().vec(), amax.sizes().vec()}};
  auto result = ::std::tuple<at::Tensor&, at::Tensor&>(out, amax);
  return hpu_op.call(result);
}

at::Tensor cast_to_fp8_q(
    const at::Tensor& input,
    at::ScalarType dtype,
    int64_t exp_bias) {
  PT_EAGER_TRACE;
  PT_OP_INFO("cast_to_fp8_q :", DUMP_3ARGS(input, dtype, exp_bias));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::cast_to_fp8_q", {input, dtype, exp_bias}, {input.sizes().vec()}};
  hpu_op.set_scalar_types({dtype});
  auto output = hpu_op.call();
  habana_helpers::set_tensor_exp_bias(output, exp_bias);
  return output;
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "cast_to_fp8_v2 :",
      DUMP_5ARGS(input, scale, stochastic_rounding, is_amax, dtype));

  TORCH_CHECK(
      dtype.has_value(),
      "hpu::cast_to_fp8_v2 without specified dtype is not available in Eager mode.");

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2",
      {input, scale, stochastic_rounding, is_amax, dtype},
      habana::CastToFp8V2OutputShape};
  hpu_op.set_scalar_types({dtype.value(), at::ScalarType::Float});
  return hpu_op.call();
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_cast_transpose(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& transposed,
    at::Tensor& amax) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_cast_transpose :",
      DUMP_6ARGS(input, scale, stochastic_rounding, out, transposed, amax));

  TORCH_CHECK(
      out.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_cast_transpose with torch.int8 dtype is not available in Eager mode.");

  habana::eager::EagerOp<std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>>
      hpu_op{
          "hpu::fp8_cast_transpose",
          {input, scale, stochastic_rounding, out, transposed, amax},
          {input.sizes().vec(), transposed.sizes().vec(), amax.sizes().vec()}};
  auto result = ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(
      out, transposed, amax);
  return hpu_op.call(result);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& transposed,
    at::Tensor& bgrad,
    at::Tensor& amax) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_cast_transpose_bgrad :",
      DUMP_7ARGS(
          input, scale, stochastic_rounding, out, transposed, bgrad, amax));

  TORCH_CHECK(
      out.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_cast_transpose_bgrad with torch.int8 dtype is not available in Eager mode.");

  habana::eager::EagerOp<
      std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>>
      hpu_op{
          "hpu::fp8_cast_transpose_bgrad",
          {input, scale, stochastic_rounding, out, transposed, bgrad, amax},
          {input.sizes().vec(),
           transposed.sizes().vec(),
           bgrad.sizes().vec(),
           amax.sizes().vec()}};
  auto result =
      ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(
          out, transposed, bgrad, amax);
  return hpu_op.call(result);
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
fp8_cast_transpose_bgrad_dgelu(
    const at::Tensor& grad,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& transposed,
    at::Tensor& bgrad,
    at::Tensor& amax) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_cast_transpose_bgrad_dgelu :",
      DUMP_9ARGS(
          grad,
          input,
          scale,
          retain,
          stochastic_rounding,
          out,
          transposed,
          bgrad,
          amax));

  TORCH_CHECK(
      out.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_cast_transpose_bgrad_dgelu with torch.int8 dtype is not available in Eager mode.");

  habana::eager::EagerOp<
      std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>>
      hpu_op{
          "hpu::fp8_cast_transpose_bgrad_dgelu",
          {grad,
           input,
           scale,
           retain,
           stochastic_rounding,
           out,
           transposed,
           bgrad,
           amax},
          {input.sizes().vec(),
           transposed.sizes().vec(),
           bgrad.sizes().vec(),
           amax.sizes().vec()}};
  auto result =
      ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(
          out, transposed, bgrad, amax);
  return hpu_op.call(result);
}

at::Tensor cast_from_fp8(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    at::ScalarType out_dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO("cast_from_fp8 :", DUMP_3ARGS(input, scale, out_dtype));

  TORCH_CHECK(
      input.scalar_type() != at::ScalarType::Char,
      "hpu::cast_from_fp8 with int8 input is not available in Eager mode.");

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::cast_from_fp8", {input, scale, out_dtype}};
  hpu_op.set_scalar_types({out_dtype});
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_dropout(
    const at::Tensor& input,
    double p,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_dropout :",
      DUMP_6ARGS(input, p, scale, stochastic_rounding, is_amax, dtype));

  TORCH_CHECK(
      dtype.has_value(),
      "hpu::fp8_dropout without specified dtype is not available in Eager mode.");

  std::vector<int64_t> amax_size{1};
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::fp8_dropout",
      {input, p, scale, stochastic_rounding, is_amax, dtype},
      {input.sizes().vec(), input.sizes().vec(), amax_size},
      0};
  hpu_op.set_scalar_types(
      {dtype.value(), c10::ScalarType::Char, c10::ScalarType::Float});
  return hpu_op.call();
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&> fp8_gelu(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& retain,
    at::Tensor& amax) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_gelu :",
      DUMP_6ARGS(input, scale, stochastic_rounding, out, retain, amax));

  TORCH_CHECK(
      out.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_cast_transpose_bgrad_dgelu with torch.int8 dtype is not available in Eager mode.");

  habana::eager::EagerOp<std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>>
      hpu_op{
          "hpu::fp8_gelu",
          {input, scale, stochastic_rounding, out, retain, amax},
          {input.sizes().vec(), input.sizes().vec(), amax.sizes().vec()}};
  auto result =
      ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&>(out, retain, amax);
  return hpu_op.call(result);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_gelu_v2(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_gelu_v2 :",
      DUMP_5ARGS(input, scale, stochastic_rounding, is_amax, dtype));

  TORCH_CHECK(
      dtype.has_value(),
      "hpu::fp8_gelu_v2 without specified dtype is not available in Eager mode.");

  std::vector<int64_t> amax_size{1};
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::fp8_gelu_v2",
      {input, scale, stochastic_rounding, is_amax, dtype},
      {input.sizes().vec(), input.sizes().vec(), amax_size},
      0};
  hpu_op.set_scalar_types(
      {dtype.value(), input.scalar_type(), c10::ScalarType::Float});
  return hpu_op.call();
}

std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> fp8_layernorm(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    at::Tensor& out,
    at::Tensor& mean,
    at::Tensor& istd,
    at::Tensor& amax) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_layernorm :",
      DUMP_10ARGS(
          input,
          weight,
          bias,
          eps,
          scale,
          stochastic_rounding,
          out,
          mean,
          istd,
          amax));

  TORCH_CHECK(
      out.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_layernorm with torch.int8 dtype is not available in Eager mode.");

  habana::eager::EagerOp<
      std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>>
      hpu_op{
          "hpu::fp8_layernorm",
          {input,
           weight,
           bias,
           eps,
           scale,
           stochastic_rounding,
           out,
           mean,
           istd,
           amax},
          {input.sizes().vec(),
           mean.sizes().vec(),
           istd.sizes().vec(),
           amax.sizes().vec()}};
  auto result =
      ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>(
          out, mean, istd, amax);
  return hpu_op.call(result);
}

at::Tensor& fp8_gemm(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const at::Tensor& D,
    at::ScalarType out_dtype,
    const c10::optional<at::Tensor>& A_scale_inv,
    const c10::optional<at::Tensor>& B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    at::Tensor& out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_gemm :",
      DUMP_11ARGS(
          A,
          trans_A,
          B,
          trans_B,
          D,
          out_dtype,
          A_scale_inv,
          B_scale_inv,
          bias,
          accumulate,
          out));

  TORCH_CHECK(
      A.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_gemm with int8 is not available in Eager mode.");

  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::fp8_gemm",
      {A,
       trans_A,
       B,
       trans_B,
       D,
       out_dtype,
       A_scale_inv,
       B_scale_inv,
       bias,
       accumulate,
       out},
      habana::Fp8GemmV2OutputShape};
  return hpu_op.call(out);
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
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_gemm_v2 :",
      DUMP_10ARGS(
          A,
          trans_A,
          B,
          trans_B,
          D,
          out_dtype,
          A_scale_inv,
          B_scale_inv,
          bias,
          accumulate));

  TORCH_CHECK(
      A.scalar_type() != at::ScalarType::Char,
      "hpu::fp8_gemm_v2 with int8 is not available in Eager mode.");

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::fp8_gemm_v2",
      {A,
       trans_A,
       B,
       trans_B,
       D,
       out_dtype,
       A_scale_inv,
       B_scale_inv,
       bias,
       accumulate},
      habana::Fp8GemmV2OutputShape};
  hpu_op.set_scalar_types({out_dtype});
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp8_bgrad_dgelu(
    const at::Tensor& grad,
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    const c10::optional<at::Tensor>& retain,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_bgrad_dgelu :",
      DUMP_7ARGS(
          grad, input, scale, retain, stochastic_rounding, is_amax, dtype));

  TORCH_CHECK(
      dtype.has_value(),
      "hpu::fp8_bgrad_dgelu without specified dtype is not available in Eager mode.");

  std::vector<int64_t> out_size = input.sizes().vec();
  std::vector<int64_t> bgrad_size{out_size[1]};
  std::vector<int64_t> amax_size{1};
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::fp8_bgrad_dgelu",
      {grad, input, scale, retain, stochastic_rounding, is_amax, dtype},
      {out_size, bgrad_size, amax_size},
      0};
  hpu_op.set_scalar_types(
      {dtype.value(), input.scalar_type(), c10::ScalarType::Float});
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> fp8_fast_softmax(
    const at::Tensor& input,
    const at::Tensor& mask,
    const c10::optional<at::Tensor>& scale,
    double softmax_scale,
    bool stochastic_rounding,
    bool is_amax,
    c10::optional<at::ScalarType> dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_fast_softmax :",
      DUMP_7ARGS(
          input,
          mask,
          scale,
          softmax_scale,
          stochastic_rounding,
          is_amax,
          dtype));

  TORCH_CHECK(
      dtype.has_value(),
      "hpu::fp8_fast_softmax without specified dtype is not available in Eager mode.");

  std::vector<int64_t> amax_size{1};
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::fp8_fast_softmax",
      {input, mask, scale, softmax_scale, stochastic_rounding, is_amax, dtype},
      {input.sizes().vec(), amax_size}};
  hpu_op.set_scalar_types({dtype.value(), c10::ScalarType::Float});
  return hpu_op.call();
}

at::Tensor fp8_reshape(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] at::IntArrayRef shape) {
  TORCH_CHECK(false, "hpu::fp8_reshape is not available in Eager mode.");
}

at::Tensor& fp8_transpose(const at::Tensor&, at::Tensor&) {
  TORCH_CHECK(false, "hpu::fp8_transpose is not available in Eager mode.");
}

at::Tensor& fp8_permute(
    [[maybe_unused]] const at::Tensor& input,
    [[maybe_unused]] at::IntArrayRef dims,
    [[maybe_unused]] at::Tensor& out) {
  TORCH_CHECK(false, "hpu::fp8_permute is not available in Eager mode.");
}

at::Tensor optimizer_lamb_norm(
    const std::vector<at::Tensor>& grad,
    double max_grad_norm) {
  PT_EAGER_TRACE;

  habana::EagerOptimizerLambNorm<at::Tensor> hpu_op{
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

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_resource_apply_momentum",
      {params_momentum_buf_list, dp_list, momentum}};

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

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_lars",
      {params, grads, skip_masks, eeta, weight_decay, eps, lr}};

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

  habana::eager::EagerOp<void> hpu_op{
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

  habana::eager::EagerOp<void> hpu_op{
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

  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_ema", {model_inputs, updated_ema, decay}};
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

  habana::eager::EagerOp<void> hpu_op{
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
      {habana::eager::eagerOpKind::Inplace, "hpu::optimizer_adamw", {1, 2, 3}});

  hpu_op.call({weight_vec, exp_avg_vec, exp_avg_sq_vec});
}

at::Tensor rotary_pos_embedding(
    const at::Tensor& input,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rotary_pos_embedding :",
      DUMP_6ARGS(input, sin, cos, position_ids, offset, mode));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::rotary_pos_embedding",
      {input, sin, cos, position_ids, offset, mode},
      {input.sizes().vec()},
      0};

  return hpu_op.call();
}

at::Tensor rotary_pos_embedding_backward(
    const at::Tensor& grad_in,
    const at::Tensor& sin,
    const at::Tensor& cos,
    const int64_t offset,
    const int64_t mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rotary_pos_embedding_backward :",
      DUMP_5ARGS(grad_in, sin, cos, offset, mode));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::rotary_pos_embedding_backward",
      {grad_in, sin, cos, offset, mode},
      {grad_in.sizes().vec()},
      0};

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> rms_norm(
    const at::Tensor& data_in,
    const at::Tensor& gamma,
    double epsilon) {
  PT_EAGER_TRACE;
  PT_OP_INFO("rms_norm :", DUMP_3ARGS(data_in, gamma, epsilon));

  std::vector<int64_t> inverse_root_mean_square_sizes{data_in.sizes().vec()};
  inverse_root_mean_square_sizes.back() = 1;

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
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
    const at::Tensor& inverse_rms,
    bool use_stages,
    int64_t bwd_mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rms_norm_backward :",
      DUMP_6ARGS(grad_in, data_in, gamma, inverse_rms, use_stages, bwd_mode));

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::rms_norm_backward",
      {grad_in, data_in, gamma, inverse_rms, use_stages, bwd_mode},
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
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "masked_batch_gemm :",
      DUMP_6ARGS(a, b, mask_a, mask_b, trans_a, trans_b));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::masked_batch_gemm",
      {a, b, mask_a, mask_b, trans_a, trans_b},
      habana::MaskedBatchGemmOutputShape};

  return hpu_op.call();
}

at::Tensor scaled_triangular_softmax(
    const at::Tensor& self,
    double inv_scale_attn,
    const c10::optional<at::Tensor>& exp_sum_recpr,
    const c10::optional<at::Tensor>& max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "scaled_triangular_softmax :",
      DUMP_4ARGS(self, inv_scale_attn, exp_sum_recpr, max));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::scaled_triangular_softmax",
      {self, inv_scale_attn, exp_sum_recpr, max},
      {self.sizes().vec()},
      0};

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> scaled_triangular_softmax_retain(
    const at::Tensor& self,
    double inv_scale_attn) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "scaled_triangular_softmax_retain :", DUMP_2ARGS(self, inv_scale_attn));

  auto out_shape = self.sizes().vec();
  auto retain_output_shape = out_shape;
  retain_output_shape.back() = 1;
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::scaled_triangular_softmax_retain",
      {self, inv_scale_attn},
      {out_shape, retain_output_shape, retain_output_shape},
      0};
  hpu_op.set_scalar_types(
      {self.scalar_type(), c10::ScalarType::Float, self.scalar_type()});

  return hpu_op.call();
}

at::Tensor& fp8_copy_(
    [[maybe_unused]] at::Tensor& self,
    [[maybe_unused]] const at::Tensor& src) {
  TORCH_CHECK(false, "hpu::fp8_copy_ is not available in Eager mode.");
}

at::Tensor& fp8_kv_reorder(
    [[maybe_unused]] at::Tensor& self,
    [[maybe_unused]] const at::Tensor& start,
    [[maybe_unused]] const at::Tensor& end,
    [[maybe_unused]] const at::Tensor& beam_idx) {
  TORCH_CHECK(false, "hpu::fp8_kv_reorder is not available in Eager mode.");
}

at::Tensor& kv_reorder_(
    at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& beam_idx) {
  PT_EAGER_TRACE;
  PT_OP_INFO("kv_reorder_ :", DUMP_4ARGS(self, start, end, beam_idx));

  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::kv_reorder_", {self, start, end, beam_idx}, {{self.sizes().vec()}}};
  return hpu_op.call(self);
}

at::Tensor kv_reorder(
    const at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& beam_idx) {
  PT_EAGER_TRACE;
  PT_OP_INFO("kv_reorder :", DUMP_4ARGS(self, start, end, beam_idx));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::kv_reorder", {self, start, end, beam_idx}, {{self.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor& fp8_index_copy_(
    [[maybe_unused]] at::Tensor& self,
    [[maybe_unused]] int64_t dim,
    [[maybe_unused]] const at::Tensor& index,
    [[maybe_unused]] const at::Tensor& source) {
  TORCH_CHECK(false, "hpu::fp8_index_copy_ is not available in Eager mode.");
}

at::Tensor fp8_repeat_v2(
    [[maybe_unused]] const at::Tensor& self,
    [[maybe_unused]] c10::SymIntArrayRef repeats) {
  TORCH_CHECK(false, "hpu::fp8_repeat_v2 is not available in Eager mode.");
}

at::Tensor fp8_index_select_v2(
    [[maybe_unused]] const at::Tensor& self,
    [[maybe_unused]] [[maybe_unused]] int64_t dim,
    [[maybe_unused]] const at::Tensor& index) {
  TORCH_CHECK(
      false, "hpu::fp8_index_select_v2 is not available in Eager mode.");
}

at::Tensor scaled_masked_triangular_softmax(
    const at::Tensor& self,
    const at::Tensor& start_end,
    double inv_scale_attn,
    int64_t grouped_batch_size,
    bool use_max,
    int64_t mode,
    c10::optional<at::ScalarType> out_dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "scaled_masked_triangular_softmax :",
      DUMP_7ARGS(
          self,
          start_end,
          inv_scale_attn,
          grouped_batch_size,
          use_max,
          mode,
          out_dtype));
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::scaled_masked_triangular_softmax",
      {self,
       start_end,
       inv_scale_attn,
       grouped_batch_size,
       use_max,
       mode,
       out_dtype},
      {{self.sizes().vec()}}};
  hpu_op.set_scalar_types({out_dtype.value_or(self.scalar_type())});
  return hpu_op.call();
}

at::Tensor& in_place_interleave_(at::Tensor& self) {
  PT_EAGER_TRACE;
  PT_OP_INFO("in_place_interleave_ :", DUMP_ARG(self));

  TORCH_CHECK(
      self.scalar_type() != at::ScalarType::Char,
      "hpu::in_place_interleave_ with int8 is not available in Eager mode.");

  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::in_place_interleave_", {self}, {{self.sizes().vec()}}};
  return hpu_op.call(self);
}

at::Tensor in_place_interleave(const at::Tensor& self) {
  PT_EAGER_TRACE;
  PT_OP_INFO("in_place_interleave :", DUMP_ARG(self));

  TORCH_CHECK(
      self.scalar_type() != at::ScalarType::Char,
      "hpu::in_place_interleave with int8 is not available in Eager mode.");

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::in_place_interleave", {self}, {{self.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor conv2d_fp8(
    const at::Tensor& input,
    const at::Tensor& weight,
    const c10::optional<at::Tensor>& bias,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    c10::optional<at::ScalarType> out_dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "conv2d_fp8 :",
      DUMP_8ARGS(
          input, weight, bias, stride, padding, dilation, groups, out_dtype));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::conv2d_fp8",
      {input, weight, bias, stride, padding, dilation, groups, out_dtype},
      habana::Conv2dFp8OutputShape};
  hpu_op.set_scalar_types({out_dtype.value_or(at::ScalarType::BFloat16)});
  return hpu_op.call();
}

at::Tensor custom_softmax(const at::Tensor& input, int64_t flavor) {
  PT_EAGER_TRACE;
  PT_OP_INFO("custom_softmax :", DUMP_2ARGS(input, flavor));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::custom_softmax", {input, flavor}, {{input.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t output_h,
    int64_t output_w,
    int64_t sampling_ratio,
    bool aligned) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "roi_align :",
      DUMP_7ARGS(
          input,
          rois,
          spatial_scale,
          output_h,
          output_w,
          sampling_ratio,
          aligned));

  std::vector<int64_t> output_shape{
      rois.size(0), input.size(1), output_h, output_w};

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "torchvision::roi_align",
      {input, rois, spatial_scale, output_h, output_w, sampling_ratio, aligned},
      {{output_shape}}};
  return hpu_op.call();
}

at::Tensor roi_align_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "_roi_align_backward :",
      DUMP_11ARGS(
          grad,
          rois,
          spatial_scale,
          pooled_height,
          pooled_width,
          batch_size,
          channels,
          height,
          width,
          sampling_ratio,
          aligned));

  std::vector<int64_t> output_shape{batch_size, channels, height, width};

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "torchvision::_roi_align_backward",
      {grad,
       rois,
       spatial_scale,
       pooled_height,
       pooled_width,
       batch_size,
       channels,
       height,
       width,
       sampling_ratio,
       aligned},
      {{output_shape}}};
  return hpu_op.call();
}
} // namespace

namespace habana::eager {

TORCH_LIBRARY(hpu, m) {
  m.def("control_edge_(Tensor(a) self)-> Tensor(a)");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor? scale, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::cast_to_fp8_q(Tensor input, ScalarType dtype, int exp_bias) -> Tensor");
  m.def(
      "hpu::cast_to_fp8_v2(Tensor input, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::conv2d_fp8(Tensor input, Tensor weight, Tensor? bias=None, int[2] stride=1, int[2] padding=0, int[2] dilation=1, int groups=1, ScalarType? out_dtype=None) -> Tensor");
  m.def("hpu::custom_softmax(Tensor input, int flavor) -> Tensor");
  m.def(
      "hpu::fp8_bgrad_dgelu(Tensor grad, Tensor input, Tensor? scale=None, Tensor? retain=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_cast_transpose(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) bgrad, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_cast_transpose_bgrad_dgelu(Tensor grad, Tensor input, Tensor? scale, Tensor? retain, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) transposed, Tensor(c!) bgrad, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def("hpu::fp8_copy_(Tensor(a!) self, Tensor src) -> Tensor(a!)");
  m.def(
      "hpu::fp8_dropout(Tensor input, float p, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_fast_softmax(Tensor input, Tensor mask, Tensor? scale, float softmax_scale, bool stochastic_rounding, bool is_amax, ScalarType? dtype=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::fp8_gelu(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) retain, Tensor(c!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!))");
  m.def(
      "hpu::fp8_gelu_v2(Tensor input, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType? dtype=None) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_gemm(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate) -> Tensor");
  m.def(
      "hpu::fp8_index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)");
  m.def(
      "hpu::fp8_index_select_v2(Tensor self, int dim, Tensor index) -> Tensor");
  m.def(
      "hpu::fp8_kv_reorder_(Tensor(a!) self, Tensor start, Tensor end, Tensor beam_idx) -> (Tensor(a!))");
  m.def(
      "hpu::fp8_layernorm(Tensor input, Tensor weight, Tensor bias, float eps, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) mean, Tensor(c!) istd, Tensor(d!) amax) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))");
  m.def(
      "hpu::fp8_permute(Tensor input, int[] dims, Tensor(a!) out) -> Tensor(a!)");
  m.def("hpu::fp8_repeat_v2(Tensor self, SymInt[] repeats) -> Tensor");
  m.def("hpu::fp8_reshape(Tensor input, int[] shape) -> Tensor");
  m.def("hpu::fp8_transpose(Tensor input, Tensor(a!) out) -> Tensor(a!)");
  m.def("hpu::in_place_interleave(Tensor self) -> Tensor");
  m.def("hpu::in_place_interleave_(Tensor(a!) self) -> (Tensor(a!))");
  m.def(
      "hpu::kv_reorder(Tensor self, Tensor start, Tensor end, Tensor beam_idx) -> Tensor");
  m.def(
      "hpu::kv_reorder_(Tensor(a!) self, Tensor start, Tensor end, Tensor beam_idx) -> (Tensor(a!))");
  m.def(
      "hpu::masked_batch_gemm(Tensor a, Tensor b, Tensor mask_a, Tensor mask_b, bool trans_a, bool trans_b) -> Tensor");
  m.def(
      "hpu::optimizer_adamw(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, Tensor weight_decay, bool has_weight_decay) -> ()");
  m.def(
      "hpu::optimizer_ema(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_fused_norm(Tensor[] grad, float max_norm) -> Tensor");
  m.def(
      "hpu::optimizer_lamb_phase1(Tensor[] gradients, Tensor[] weights, Tensor(a!)[] exp_avg, Tensor(b!)[] exp_avg_sq, Tensor(c!)[] out_weight_norms, Tensor(d!)[] out_adam_norms, Tensor(e!)[] out_adam_steps, Tensor clip_global_grad_norm, int grad_averaging, float beta1, float beta2, float epsilon, Tensor bias_correction1, Tensor bias_correction2, float weight_decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_phase2(Tensor(a!)[] weights, Tensor[] adam_norms, Tensor[] weight_norms, Tensor[] adam_steps, Tensor neg_step, float wd, bool use_lamb) -> ()");
  m.def(
      "hpu::optimizer_lars(Tensor[] params, Tensor(a!)[] grads, int[] skip_masks, float eeta, float weight_decay, float eps, Tensor lr) -> ()");
  m.def(
      "hpu::optimizer_resource_apply_momentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::rms_norm(Tensor data_in, Tensor gamma, float epsilon) -> (Tensor, Tensor)");
  m.def(
      "hpu::rms_norm_backward(Tensor grad_in, Tensor data_in, Tensor gamma, Tensor inverse_rms, bool use_stages, int bwd_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::rotary_pos_embedding(Tensor input, Tensor sin, Tensor cos, Tensor? position_ids, int offset, int mode) -> Tensor");
  m.def(
      "hpu::rotary_pos_embedding_backward(Tensor grad_in, Tensor sin, Tensor cos, int offset, int mode) -> Tensor");
  m.def(
      "hpu::scaled_masked_triangular_softmax(Tensor self, Tensor start_end, float inv_scale_attn, int grouped_batch_size, bool use_max, int mode, ScalarType? out_dtype=None) -> Tensor");
  m.def(
      "hpu::scaled_triangular_softmax(Tensor self, float inv_scale_attn, Tensor? exp_sum_recpr=None, Tensor? max=None) -> Tensor");
  m.def(
      "hpu::scaled_triangular_softmax_retain(Tensor self, float inv_scale_attn) -> (Tensor, Tensor, Tensor)");
  m.def("hpu::view(Tensor input, Tensor shape) -> Tensor");
  m.def("hpu::view_neg(Tensor input, Tensor shape, int[] shape) -> Tensor");
  m.def(
      "strided_insert_orig_ds(Tensor self, Tensor other, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_orig_ds_h2d(Tensor self, Tensor other, Tensor stride) -> (Tensor)");
  m.def(
      "strided_view_ds_h2d(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_view_orig_ds_h2d(Tensor self, Tensor size, Tensor stride) -> (Tensor)");
  m.def("hpu::habana_bernoulli(Tensor self, Tensor seed) -> Tensor");
  m.def(
      "hpu::habana_rand(SymInt[] size, Tensor seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_randn(SymInt[] size, Tensor seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_seed_generator(Tensor seed, Tensor counter, int size) -> Tensor");
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::cast_from_fp8", cast_from_fp8);
  m.impl("hpu::cast_to_fp8", cast_to_fp8);
  m.impl("hpu::cast_to_fp8_q", cast_to_fp8_q);
  m.impl("hpu::cast_to_fp8_v2", cast_to_fp8_v2);
  m.impl("hpu::conv2d_fp8", conv2d_fp8);
  m.impl("hpu::custom_softmax", custom_softmax);
  m.impl("hpu::fp8_bgrad_dgelu", fp8_bgrad_dgelu);
  m.impl("hpu::fp8_cast_transpose", fp8_cast_transpose);
  m.impl("hpu::fp8_cast_transpose_bgrad", fp8_cast_transpose_bgrad);
  m.impl("hpu::fp8_cast_transpose_bgrad_dgelu", fp8_cast_transpose_bgrad_dgelu);
  m.impl("hpu::fp8_copy_", fp8_copy_);
  m.impl("hpu::fp8_dropout", fp8_dropout);
  m.impl("hpu::fp8_fast_softmax", fp8_fast_softmax);
  m.impl("hpu::fp8_gelu", fp8_gelu);
  m.impl("hpu::fp8_gelu_v2", fp8_gelu_v2);
  m.impl("hpu::fp8_gemm", fp8_gemm);
  m.impl("hpu::fp8_gemm_v2", fp8_gemm_v2);
  m.impl("hpu::fp8_index_copy_", fp8_index_copy_);
  m.impl("hpu::fp8_index_select_v2", fp8_index_select_v2);
  m.impl("hpu::fp8_kv_reorder_", fp8_kv_reorder);
  m.impl("hpu::fp8_layernorm", fp8_layernorm);
  m.impl("hpu::fp8_permute", fp8_permute);
  m.impl("hpu::fp8_repeat_v2", fp8_repeat_v2);
  m.impl("hpu::fp8_reshape", fp8_reshape);
  m.impl("hpu::fp8_transpose", fp8_transpose);
  m.impl("hpu::in_place_interleave", in_place_interleave);
  m.impl("hpu::in_place_interleave_", in_place_interleave_);
  m.impl("hpu::kv_reorder", kv_reorder);
  m.impl("hpu::kv_reorder_", kv_reorder_);
  m.impl("hpu::masked_batch_gemm", masked_batch_gemm);
  m.impl("hpu::optimizer_adamw", optimizer_adamw);
  m.impl("hpu::optimizer_ema", optimizer_ema);
  m.impl("hpu::optimizer_lamb_fused_norm", optimizer_lamb_norm);
  m.impl("hpu::optimizer_lamb_phase1", optimizer_lamb_phase1);
  m.impl("hpu::optimizer_lamb_phase2", optimizer_lamb_phase2);
  m.impl("hpu::optimizer_lars", optimizer_lars);
  m.impl(
      "hpu::optimizer_resource_apply_momentum",
      optimizer_resource_apply_momentum);
  m.impl("hpu::rms_norm", rms_norm);
  m.impl("hpu::rms_norm_backward", rms_norm_backward);
  m.impl("hpu::rotary_pos_embedding", rotary_pos_embedding);
  m.impl("hpu::rotary_pos_embedding_backward", rotary_pos_embedding_backward);
  m.impl(
      "hpu::scaled_masked_triangular_softmax",
      scaled_masked_triangular_softmax);
  m.impl("hpu::scaled_triangular_softmax", scaled_triangular_softmax);
  m.impl(
      "hpu::scaled_triangular_softmax_retain",
      scaled_triangular_softmax_retain);
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl("roi_align", roi_align);
  m.impl("_roi_align_backward", roi_align_backward);
}
} // namespace habana::eager

namespace {
// Inplace ops must be additionally registered to Functionalize backend
// to be handled in torch.compile
// https://gist.github.com/bdhirsh/7dadbf6296f8f7d1abcf4c482f438aaa
at::Tensor get_functional_tensor(const at::Tensor& tensor) {
  TORCH_INTERNAL_ASSERT(
      at::functionalization::impl::isFunctionalTensor(tensor));
  at::functionalization::impl::sync(tensor);
  return at::functionalization::impl::from_functional_tensor(tensor);
}

at::Tensor& kv_reorder_functionalization_glue(
    at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& beam_idx) {
  auto self_ = get_functional_tensor(self);
  auto start_ = get_functional_tensor(start);
  auto end_ = get_functional_tensor(end);
  auto beam_idx_ = get_functional_tensor(beam_idx);

  static auto op_handle = c10::Dispatcher::singleton()
                              .findSchemaOrThrow("hpu::kv_reorder", "")
                              .typed<at::Tensor(
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  const at::Tensor&,
                                  const at::Tensor&)>();

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = op_handle.call(self_, start_, end_, beam_idx_);
  }

  at::functionalization::impl::replace_(self, tmp_output);
  at::functionalization::impl::commit_update(self);
  at::functionalization::impl::sync(self);
  return self;
}

at::Tensor& in_place_interleave_functionalization_glue(at::Tensor& self) {
  auto self_ = get_functional_tensor(self);

  static auto op_handle = c10::Dispatcher::singleton()
                              .findSchemaOrThrow("hpu::in_place_interleave", "")
                              .typed<at::Tensor(const at::Tensor&)>();

  at::Tensor tmp_output;
  {
    at::AutoDispatchSkipFunctionalize guard;
    tmp_output = op_handle.call(self_);
  }

  at::functionalization::impl::replace_(self, tmp_output);
  at::functionalization::impl::commit_update(self);
  at::functionalization::impl::sync(self);
  return self;
}
} // namespace

namespace habana::eager {

TORCH_LIBRARY_IMPL(hpu, Functionalize, m) {
  m.impl("kv_reorder_", kv_reorder_functionalization_glue);
  m.impl("in_place_interleave_", in_place_interleave_functionalization_glue);
}

} // namespace habana::eager
