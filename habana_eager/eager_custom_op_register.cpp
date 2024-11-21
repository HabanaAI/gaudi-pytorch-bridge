/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <ATen/ATen.h>
#include <ATen/FunctionalTensorWrapper.h>
#include <ATen/Tensor.h>
#include <torch/library.h>
#include "backend/random.h"
#include "common/dump_args.h"
#include "common/random_utils.h"
#include "habana_eager/graph_weight_permute.h"
#include "habana_eager/ops/eager_op.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/ctc_loss_custom.h"
#include "hpu_ops/fp8_ops.h"
#include "hpu_ops/masked_batch_gemm.h"
#include "hpu_ops/op_logger.h"
#include "hpu_ops/optimizer_lamb_gen.h"
#include "hpu_ops/sdpa_gen.h"

namespace {
using habana::to_string; // For DUMP_*ARGS

/***********************************************************************************
 * Custom ops
 **********************************************************************************/

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

  habana::eager::EagerOp<std::tuple<at::Tensor&, at::Tensor&>> hpu_op{
      "hpu::cast_to_fp8",
      {input, scale, stochastic_rounding, out, amax},
      {input.sizes().vec(), amax.sizes().vec()}};
  auto result = ::std::tuple<at::Tensor&, at::Tensor&>(out, amax);
  return hpu_op.call(result);
}

template <class T>
static std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_common(
    const at::Tensor& input,
    T scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "cast_to_fp8_v2 :",
      DUMP_6ARGS(
          input, scale, stochastic_rounding, is_amax, dtype, scale_shape));

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2",
      {input, scale, stochastic_rounding, is_amax, dtype, scale_shape},
      habana::CastToFp8V2OutputShape};
  hpu_op.set_scalar_types({dtype, at::ScalarType::Float});
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_to_fp8_v2_common(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar(
    const at::Tensor& input,
    double scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef) {
  return cast_to_fp8_v2_common(
      input, scale, stochastic_rounding, is_amax, dtype, c10::nullopt);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar_list(
    const at::Tensor& input,
    c10::ArrayRef<double> scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_to_fp8_v2_common(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
}

template <class T>
static at::Tensor cast_from_fp8_common(
    const at::Tensor& input,
    T scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "cast_from_fp8 :", DUMP_4ARGS(input, scale, out_dtype, scale_shape));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::cast_from_fp8", {input, scale, out_dtype, scale_shape}};
  hpu_op.set_scalar_types({out_dtype});
  return hpu_op.call();
}

at::Tensor cast_from_fp8(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_from_fp8_common(input, scale, out_dtype, scale_shape);
}

at::Tensor cast_from_fp8_scalar(
    const at::Tensor& input,
    double scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef) {
  return cast_from_fp8_common(input, scale, out_dtype, c10::nullopt);
}

at::Tensor cast_from_fp8_scalar_list(
    const at::Tensor& input,
    c10::ArrayRef<double> scale,
    at::ScalarType out_dtype,
    OptionalIntArrayRef scale_shape) {
  return cast_from_fp8_common(input, scale, out_dtype, scale_shape);
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

template <class T>
static at::Tensor fp8_gemm_v2_common(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    T A_scale_inv,
    T B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_gemm_v2 :",
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
          B_scale_shape));

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
       accumulate,
       B_scale_shape},
      habana::Fp8GemmV2OutputShape};
  hpu_op.set_scalar_types({out_dtype});
  return hpu_op.call();
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
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  return fp8_gemm_v2_common(
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
      B_scale_shape);
}

at::Tensor fp8_gemm_v2_scalar(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    double A_scale_inv,
    double B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef) {
  return fp8_gemm_v2_common(
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
      c10::nullopt);
}

at::Tensor fp8_gemm_v2_scalar_list(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const c10::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    c10::ArrayRef<double> A_scale_inv,
    c10::ArrayRef<double> B_scale_inv,
    const c10::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  return fp8_gemm_v2_common(
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
      B_scale_shape);
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

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_lamb_phase1",
       decltype(habana::eager::EagerOpMetaData::out_indices_){2, 3, 4, 5, 6}});

  std::vector<at::TensorList> tensorlists = {
      exp_avg, exp_avg_sq, out_weight_norms, out_adam_norms, out_adam_steps};

  return hpu_op.call(tensorlists);
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
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_lamb_phase2",
       decltype(habana::eager::EagerOpMetaData::out_indices_){0}});
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
    const bool has_weight_decay,
    c10::optional<at::TensorList> exp_avg_scales = c10::nullopt,
    c10::optional<at::TensorList> exp_avg_sq_scales = c10::nullopt) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "optimizer_adamw :",
      DUMP_12ARGS(
          gradient_vec,
          weight_vec,
          exp_avg_vec,
          exp_avg_sq_vec,
          neg_step_t,
          beta1,
          beta2,
          epsilon,
          weight_decay,
          has_weight_decay,
          exp_avg_scales,
          exp_avg_sq_scales));

  TORCH_CHECK(
      (weight_vec.size() > 0),
      "optimizer_adamw : can not process empty weight vector");
  TORCH_CHECK(
      exp_avg_scales.has_value() == exp_avg_sq_scales.has_value(),
      "optimizer_adamw : expects both or neighter scales to be set");

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
       has_weight_decay,
       exp_avg_scales,
       exp_avg_sq_scales}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_adamw",
       {1, 2, 3, 10, 11}});

  std::vector<at::TensorList> tensorlists = {
      weight_vec, exp_avg_vec, exp_avg_sq_vec};
  if (exp_avg_scales.has_value()) {
    tensorlists.push_back(exp_avg_scales.value());
    tensorlists.push_back(exp_avg_sq_scales.value());
  }
  hpu_op.call(tensorlists);
}

at::Tensor fused_clip_norm(
    at::TensorList grad,
    const at::Tensor& max_norm,
    double norm_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO("fused_clip_norm :", DUMP_3ARGS(grad, max_norm, norm_type));

  TORCH_CHECK(
      (grad.size() > 0),
      "fused_clip_norm : can not process empty grad vector (eager)");

  habana::eager::EagerOp<void> hpu_op{
      "hpu::fused_clip_norm", {grad, max_norm, norm_type}};

  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::fused_clip_norm",
       decltype(habana::eager::EagerOpMetaData::out_indices_){0}});

  hpu_op.call(grad);

  // return the total_norm result from the end of the grad vector
  return grad.back();
}

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_common(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_common :",
      DUMP_11ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode));
  // experts_min/max are used by CGuid path only,
  // so they don't affect eager execution

  std::function<at::Tensor(const at::Tensor& x)> activation_fn;
  if (activation == "gelu") {
    activation_fn = [](const at::Tensor& x) {
      return torch::nn::functional::gelu(x);
    };
  } else if (activation == "relu") {
    activation_fn = [](const at::Tensor& x) {
      return torch::nn::functional::relu(x);
    };
  } else if (activation == "silu") {
    activation_fn = [](const at::Tensor& x) {
      return torch::nn::functional::silu(x);
    };
  }
  const int num_experts = w1.size();
  const int num_tokens = hidden_states.size(0);
  const int hidden_dim = hidden_states.size(1);
  auto final_hidden_states =
      torch::zeros({1, num_tokens, hidden_dim}, hidden_states.options());
  auto padded_weights =
      torch::zeros({num_tokens, num_experts}, hidden_states.options())
          .scatter_(-1, expert_routing_table, router_weights)
          .reshape({-1, num_tokens, num_experts})
          .permute({2, 0, 1})
          .unsqueeze(-1);

  auto amax_per_expert =
      torch::zeros({num_experts}, torch::dtype(torch::kFloat32));
  for (int expert_idx = 0; expert_idx < num_experts; expert_idx++) {
    const at::Tensor current_expert_w1 =
        permuted_weights ? w1[expert_idx].transpose(0, 1) : w1[expert_idx];
    const at::Tensor current_expert_w2 =
        permuted_weights ? w2[expert_idx].transpose(0, 1) : w2[expert_idx];
    const at::Tensor current_expert_w3 =
        permuted_weights ? w3[expert_idx].transpose(0, 1) : w3[expert_idx];

    auto hidden_states_w1 =
        activation_fn(torch::matmul(hidden_states, current_expert_w1));
    auto hidden_states_w2 = torch::matmul(hidden_states, current_expert_w2);
    auto hidden_states_w12 = hidden_states_w1 * hidden_states_w2;
    amax_per_expert[expert_idx] =
        torch::amax(hidden_states_w12).to(torch::kFloat32);
    auto hidden_states_w3 = torch::matmul(hidden_states_w12, current_expert_w3);
    final_hidden_states += hidden_states_w3 * padded_weights[expert_idx];
  }
  auto result = final_hidden_states.reshape(hidden_states.sizes());
  return std::make_tuple(result, amax_per_expert);
}

at::Tensor mixture_of_experts(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  auto moe_common = mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      false);

  return std::get<0>(moe_common);
}

at::Tensor mixture_of_experts_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fused_weights :",
      DUMP_9ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  std::vector<at::Tensor> w1, w2;
  const auto splitDim = permuted_weights ? 0 : 1;
  const auto splitIndex = w12[0].size(splitDim) / 2;
  for (const auto& tensor : w12) {
    auto w12_split = tensor.split(splitIndex, splitDim);
    w1.push_back(w12_split[0]);
    w2.push_back(w12_split[1]);
  }

  auto moe_common = mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      false);
  return std::get<0>(moe_common);
}

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_measurement :",
      DUMP_11ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode));
  return mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      measurement_mode);
}

std::tuple<at::Tensor, at::Tensor>
mixture_of_experts_fp8_measurement_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_measurement_fused_weights :",
      DUMP_10ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max,
          measurement_mode));

  std::vector<at::Tensor> w1, w2;
  const auto splitDim = permuted_weights ? 0 : 1;
  const auto splitIndex = w12[0].size(splitDim) / 2;
  for (const auto& tensor : w12) {
    auto w12_split = tensor.split(splitIndex, splitDim);
    w1.push_back(w12_split[0]);
    w2.push_back(w12_split[1]);
  }

  return mixture_of_experts_common(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max,
      measurement_mode);
}

void optimizer_sgd(
    const at::TensorList gradients,
    at::TensorList weights,
    at::Tensor& lr,
    double wd,
    double mom,
    double damp,
    bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd:",
      DUMP_7ARGS(gradients, weights, lr, wd, mom, damp, nesterov));
  TORCH_CHECK(
      (weights.size() > 0),
      "optimizer_sgd : can not process empty weight vector");
  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd", {gradients, weights, lr, wd, mom, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_sgd",
       decltype(habana::eager::EagerOpMetaData::out_indices_){1}});
  hpu_op.call({weights});
}

void optimizer_sgd_momentum(
    const at::TensorList gradients,
    at::TensorList weights,
    at::TensorList momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    at::Tensor& mom,
    double wd,
    double damp,
    bool nesterov) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      " optimizer_sgd_momentum:",
      DUMP_9ARGS(
          gradients,
          weights,
          momentum,
          epoch_num,
          lr,
          mom,
          wd,
          damp,
          nesterov));
  TORCH_CHECK(
      (weights.size() > 0),
      "optimizer_sgd_momentum : can not process empty weight vector");
  habana::eager::EagerOp<void> hpu_op{
      "hpu::optimizer_sgd_momentum",
      {gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::optimizer_sgd_momentum",
       decltype(habana::eager::EagerOpMetaData::out_indices_){1, 2}});
  hpu_op.call({weights, momentum});
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
    const c10::optional<at::Tensor>& position_ids,
    const int64_t offset,
    const int64_t mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "rotary_pos_embedding_backward :",
      DUMP_6ARGS(grad_in, sin, cos, position_ids, offset, mode));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::rotary_pos_embedding_backward",
      {grad_in, sin, cos, position_ids, offset, mode},
      {grad_in.sizes().vec()},
      0};

  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor> ctc_loss_custom(
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "ctc_loss_custom :",
      DUMP_7ARGS(
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          blank,
          reduction,
          zero_infinity));
  auto shapes = habana::calculate_output_shapes_for_ctc_loss_custom_fwd(
      log_probs, targets, reduction);

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::ctc_loss_custom",
      {log_probs,
       targets,
       input_lengths,
       target_lengths,
       blank,
       reduction,
       zero_infinity},
      {std::get<0>(shapes), std::get<1>(shapes)},
      0};

  return hpu_op.call();
}

at::Tensor ctc_loss_custom_backward(
    const at::Tensor& grad,
    const at::Tensor& log_probs,
    const at::Tensor& targets,
    const at::Tensor& input_lengths,
    const at::Tensor& target_lengths,
    const at::Tensor& neg_log_likelihood,
    const at::Tensor& log_alpha,
    int64_t blank,
    int64_t reduction,
    bool zero_infinity) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "ctc_loss_custom_backward :",
      DUMP_10ARGS(
          grad,
          log_probs,
          targets,
          input_lengths,
          target_lengths,
          neg_log_likelihood,
          log_alpha,
          blank,
          reduction,
          zero_infinity));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::ctc_loss_custom_backward",
      {grad,
       log_probs,
       targets,
       input_lengths,
       target_lengths,
       neg_log_likelihood,
       log_alpha,
       blank,
       reduction,
       zero_infinity},
      {log_probs.sizes().vec()},
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

at::Tensor& kv_reorder_(
    at::Tensor& self,
    const at::Tensor& start,
    const at::Tensor& end,
    const at::Tensor& beam_idx) {
  PT_EAGER_TRACE;
  PT_OP_INFO("kv_reorder_ :", DUMP_4ARGS(self, start, end, beam_idx));

  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::kv_reorder_", {self, start, end, beam_idx}, {{self.sizes().vec()}}};
  hpu_op.set_eager_op_info(
      {habana::eager::eagerOpKind::Inplace,
       "hpu::kv_reorder_",
       decltype(habana::eager::EagerOpMetaData::out_indices_){0}});
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

at::Tensor _ragged_softmax(
    const at::Tensor& self,
    int64_t dim,
    bool half_to_float,
    const at::Tensor& valid_count) {
  PT_EAGER_TRACE;

  PT_OP_INFO(
      "HpuOp _ragged_softmax :",
      " self=",
      to_string(self),
      " dim=",
      to_string(dim),
      " half_to_float=",
      to_string(half_to_float));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::ragged_softmax", {self, dim, half_to_float, valid_count}};
  return hpu_op.call();
}

at::Tensor scaled_masked_softmax(
    const at::Tensor& input,
    const at::Tensor& mask,
    double scale) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "scaled_masked_softmax:",
      " input=",
      to_string(input),
      " mask=",
      to_string(mask),
      " scale=",
      to_string(scale));
  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::scaled_masked_softmax",
      {input, mask, scale},
      {{input.sizes().vec()}}};
  return hpu_op.call();
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

  habana::eager::EagerOp<at::Tensor&> hpu_op{
      "hpu::in_place_interleave_", {self}, {{self.sizes().vec()}}};
  return hpu_op.call(self);
}

at::Tensor in_place_interleave(const at::Tensor& self) {
  PT_EAGER_TRACE;
  PT_OP_INFO("in_place_interleave :", DUMP_ARG(self));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::in_place_interleave", {self}, {{self.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor custom_softmax(const at::Tensor& input, int64_t flavor) {
  PT_EAGER_TRACE;
  PT_OP_INFO("custom_softmax :", DUMP_2ARGS(input, flavor));

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::custom_softmax", {input, flavor}, {{input.sizes().vec()}}};
  return hpu_op.call();
}

at::Tensor slice_ds(
    const at::Tensor& self,
    c10::SymInt dim,
    c10::SymInt start,
    c10::SymInt end,
    c10::SymInt step,
    [[maybe_unused]] c10::optional<c10::SymIntArrayRef> size) {
  PT_EAGER_TRACE;
  PT_OP_INFO("slice_ds :", DUMP_5ARGS(self, size, dim, start, end));
  return at::native::slice(
      self,
      dim.expect_int(),
      start.expect_int(),
      end.expect_int(),
      step.expect_int());
}

at::Tensor constant_pad_nd_ds(
    const at::Tensor& self,
    c10::SymIntArrayRef pad,
    const c10::Scalar& value,
    [[maybe_unused]] c10::optional<c10::SymIntArrayRef> size) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "constant_pad_ds :",
      DUMP_3ARGS(self, c10::asIntArrayRefUnchecked(pad), value));
  return at::native::constant_pad_nd(
      self, c10::asIntArrayRefUnchecked(pad), value);
}

// accumulate_grads_ is a wrapper for native inductor.accumulate_grad_ op.
// It extracts gradients from variables and assigns respective new_grads to them
// or increment by them, depending if gradients are defined.
void accumulate_grads_(
    at::TensorList variables,
    const at::TensorList new_grads) {
  PT_EAGER_TRACE;
  PT_OP_INFO("accumulate_grads_ :", DUMP_2ARGS(variables, new_grads));

  TORCH_CHECK(
      variables.size() == new_grads.size(),
      "Inputs to hpu::accumulate_grads_ must be of the same size, got: ",
      variables.size(),
      " and ",
      new_grads.size());

  if (variables.empty()) {
    PT_BRIDGE_WARN("hpu::accumulate_grads_ received empty inputs.");
    return;
  }

  if (variables[0].mutable_grad().defined()) {
    std::vector<at::Tensor> current_grads_list;
    current_grads_list.reserve(variables.size());
    for (auto& variable : variables) {
      current_grads_list.push_back(variable.mutable_grad());
    }
    habana::eager::EagerOp<void> hpu_op{
        "hpu::custom_foreach_add_", {current_grads_list, new_grads}};
    hpu_op.call(current_grads_list);
  } else {
    for (size_t i = 0; i < variables.size(); ++i) {
      variables[i].mutable_grad() = new_grads[i];
    }
  }
}

at::Tensor convert_from_int4_common(
    const std::string& op_name,
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype) {
  PT_EAGER_TRACE;
  PT_OP_INFO(op_name + " :", DUMP_4ARGS(input, scale, zero_point, out_dtype));

  auto output_shape = input.sizes().vec();
  output_shape.back() *= 8;

  habana::eager::EagerOp<at::Tensor> hpu_op{
      "hpu::" + op_name, {input, scale, zero_point, out_dtype}, {output_shape}};
  hpu_op.set_scalar_types({out_dtype});
  return hpu_op.call();
}

at::Tensor convert_from_int4(
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype) {
  return convert_from_int4_common(
      "convert_from_int4", input, scale, zero_point, out_dtype);
}

at::Tensor convert_from_uint4(
    const at::Tensor& input,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& zero_point,
    at::ScalarType out_dtype) {
  return convert_from_int4_common(
      "convert_from_uint4", input, scale, zero_point, out_dtype);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_recomp_fwd :",
      DUMP_11ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          requires_backward,
          softmax_mode,
          valid_seq_len,
          seq_padding_type));

  if (p > 0.0) {
    int seed = habana::get_seed_hpu(c10::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_recomp_fwd_dropout_seed",
            {seed_t,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(),
         q.scalar_type(),
         c10::ScalarType::Float,
         c10::ScalarType::Int});
    return hpu_op.call();

  } else {
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_recomp_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPARecompFwdOutputShape};
    auto linvType = c10::ScalarType::Float;

    if ((softmax_mode == "fast") &&
        (q.scalar_type() == c10::ScalarType::BFloat16)) {
      linvType = c10::ScalarType::BFloat16;
    }
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), linvType, c10::ScalarType::Int});
    return hpu_op.call();
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_recomp_bwd(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const at::Tensor& m,
    const at::Tensor& linv,
    const c10::optional<at::Tensor>& seed,
    const bool is_causal,
    const double p,
    const double scale,
    c10::string_view softmax_mode,
    const at::Tensor& fwd_out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_recomp_bwd :",
      DUMP_13ARGS(
          grad,
          q,
          k,
          v,
          attention_mask,
          m,
          linv,
          seed,
          is_causal,
          p,
          scale,
          softmax_mode,
          fwd_out));

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::sdpa_recomp_bwd",
      {grad,
       q,
       k,
       v,
       attention_mask,
       m,
       linv,
       seed,
       is_causal,
       p,
       scale,
       softmax_mode,
       fwd_out},
      habana::SDPARecompBwdOutputShape};
  return hpu_op.call();
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_fwd :",
      DUMP_10ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          softmax_mode,
          valid_seq_len,
          seq_padding_type));
  if (p > 0.0) {
    int seed = habana::get_seed_hpu(c10::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_fwd_dropout_seed",
            {seed_t,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), c10::ScalarType::Char});
    return hpu_op.call();

  } else {
    habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::sdpa_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             valid_seq_len,
             seq_padding_type},
            habana::SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {q.scalar_type(), q.scalar_type(), c10::ScalarType::Char});
    return hpu_op.call();
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> sdpa_bwd(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const at::Tensor& fwd_out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "sdpa_bwd :",
      DUMP_10ARGS(grad, q, k, v, P, dm, is_causal, p, scale, fwd_out));
  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor, at::Tensor>> hpu_op{
      "hpu::sdpa_bwd",
      {grad, q, k, v, P, dm, is_causal, p, scale, fwd_out},
      habana::SDPABwdOutputShape};

  return hpu_op.call();
}

at::Tensor weight_permutation(const at::Tensor& weight) {
  habana::graph::PermuteWeightTensor t(weight);
  t.PermuteIfNeeded();
  return weight;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_o,
    const c10::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_fwd :",
      DUMP_17ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          q_scale_s,
          q_scale_o,
          d_scale_s,
          is_amax_s,
          valid_seq_len,
          seq_padding_type));
  auto fwdOutType = q.scalar_type();
  auto sfmxType = q.scalar_type();
  // if (is_amax_s) {
  // sfmxType = c10::ScalarType::BFloat16;
  // }

  // Normally SDPA FWD Fp8 dtype is e4m3. Supporting
  // e5m2 for experiments
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn ||
      q.scalar_type() == at::ScalarType::Float8_e5m2) {
    if (q_scale_o.has_value()) {
      fwdOutType = q.scalar_type();
    } else {
      fwdOutType = at::ScalarType::BFloat16;
    }
  }

  if (p > 0.0) {
    int seed = habana::get_seed_hpu(c10::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_fwd_dropout_seed",
            {seed,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType, sfmxType, c10::ScalarType::Char, c10::ScalarType::Float});

    return hpu_op.call();

  } else {
    habana::eager::EagerOp<
        std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPAFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType, sfmxType, c10::ScalarType::Char, c10::ScalarType::Float});
    return hpu_op.call();
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp8_sdpa_bwd(
    const at::Tensor& grad,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& P,
    const c10::optional<at::Tensor>& dm,
    const bool is_causal,
    const double p,
    const double scale,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& d_scale_s,
    const c10::optional<at::Tensor>& d_scale_do,
    const c10::optional<at::Tensor>& d_scale_ds,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_ds,
    const bool is_amax_ds,
    const at::Tensor& fwd_out) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_bwd :",
      DUMP_19ARGS(
          grad,
          q,
          k,
          v,
          P,
          dm,
          is_causal,
          p,
          scale,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          d_scale_s,
          d_scale_do,
          d_scale_ds,
          q_scale_s,
          q_scale_ds,
          is_amax_ds,
          fwd_out));

  habana::eager::EagerOp<
      std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>>
      hpu_op{
          "hpu::fp8_sdpa_bwd",
          {grad,
           q,
           k,
           v,
           P,
           dm,
           is_causal,
           p,
           scale,
           d_scale_q,
           d_scale_k,
           d_scale_v,
           d_scale_s,
           d_scale_do,
           d_scale_ds,
           q_scale_s,
           q_scale_ds,
           is_amax_ds,
           fwd_out},
          habana::Fp8SDPABwdOutputShape};

  // Set grad type to BF16 for now
  auto gradType = c10::ScalarType::BFloat16;
  hpu_op.set_scalar_types(
      {gradType, // dQ
       gradType, // dK
       gradType, // dV
       c10::ScalarType::Float}); // amax_ds

  return hpu_op.call();
}

template <class T>
static std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd_common(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    T d_scale_q,
    T d_scale_k,
    T d_scale_v,
    T q_scale_s,
    T q_scale_o,
    T d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type,
    c10::ScalarType fwdOutType) {
  PT_EAGER_TRACE;
  PT_OP_INFO(
      "fp8_sdpa_recomp_fwd :",
      DUMP_19ARGS(
          q,
          k,
          v,
          attention_mask,
          p,
          scale,
          is_causal,
          requires_backward,
          softmax_mode,
          d_scale_q,
          d_scale_k,
          d_scale_v,
          q_scale_s,
          q_scale_o,
          d_scale_s,
          is_amax_s,
          is_amax_o,
          valid_seq_len,
          seq_padding_type));

  auto linvType = c10::ScalarType::Float;

  if ((softmax_mode == "fast") &&
      (q.scalar_type() == c10::ScalarType::BFloat16)) {
    linvType = c10::ScalarType::BFloat16;
  }
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    linvType = c10::ScalarType::BFloat16;
  }

  auto mType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn) {
    mType = c10::ScalarType::BFloat16;
  }

  if (p > 0.0) {
    int seed = habana::get_seed_hpu(c10::nullopt);
    at::TensorOptions o;
    o = o.dtype(at::kInt).device(at::kHPU);
    at::Tensor seed_t = at::tensor(seed, o);
    habana::eager::EagerOp<std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_recomp_fwd_dropout_seed",
            {seed,
             q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             is_amax_o,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType,
         mType,
         linvType,
         c10::ScalarType::Int,
         c10::ScalarType::Float,
         c10::ScalarType::Float});

    return hpu_op.call();
  } else {
    habana::eager::EagerOp<std::tuple<
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor,
        at::Tensor>>
        hpu_op{
            "hpu::fp8_sdpa_recomp_fwd",
            {q,
             k,
             v,
             attention_mask,
             p,
             scale,
             is_causal,
             requires_backward,
             softmax_mode,
             d_scale_q,
             d_scale_k,
             d_scale_v,
             q_scale_s,
             q_scale_o,
             d_scale_s,
             is_amax_s,
             is_amax_o,
             valid_seq_len,
             seq_padding_type},
            habana::Fp8SDPARecompFwdOutputShape};
    hpu_op.set_scalar_types(
        {fwdOutType,
         mType,
         linvType,
         c10::ScalarType::Int,
         c10::ScalarType::Float,
         c10::ScalarType::Float});

    return hpu_op.call();
  }
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const c10::optional<at::Tensor>& d_scale_q,
    const c10::optional<at::Tensor>& d_scale_k,
    const c10::optional<at::Tensor>& d_scale_v,
    const c10::optional<at::Tensor>& q_scale_s,
    const c10::optional<at::Tensor>& q_scale_o,
    const c10::optional<at::Tensor>& d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  auto fwdOutType = q.scalar_type();

  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn &&
      (!q_scale_o.has_value()))
    fwdOutType = at::ScalarType::BFloat16;
  return fp8_sdpa_recomp_fwd_common<c10::optional<at::Tensor>>(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      is_amax_o,
      valid_seq_len,
      seq_padding_type,
      fwdOutType);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor>
fp8_sdpa_recomp_scalar_fwd(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const c10::optional<at::Tensor>& attention_mask,
    const double p,
    const double scale,
    const bool is_causal,
    const bool requires_backward,
    c10::string_view softmax_mode,
    const double d_scale_q,
    const double d_scale_k,
    const double d_scale_v,
    const double q_scale_s,
    const double q_scale_o,
    const double d_scale_s,
    const bool is_amax_s,
    const bool is_amax_o,
    const c10::optional<at::Tensor>& valid_seq_len,
    c10::string_view seq_padding_type) {
  PT_EAGER_TRACE;
  auto fwdOutType = q.scalar_type();
  if (q.scalar_type() == at::ScalarType::Float8_e4m3fn && (q_scale_o == 0.))
    fwdOutType = at::ScalarType::BFloat16;

  return fp8_sdpa_recomp_fwd_common<double>(
      q,
      k,
      v,
      attention_mask,
      p,
      scale,
      is_causal,
      requires_backward,
      softmax_mode,
      d_scale_q,
      d_scale_k,
      d_scale_v,
      q_scale_s,
      q_scale_o,
      d_scale_s,
      is_amax_s,
      is_amax_o,
      valid_seq_len,
      seq_padding_type,
      fwdOutType);
}

/***********************************************************************************
 * Native ops
 **********************************************************************************/

at::Tensor nms(
    const at::Tensor& boxes,
    const at::Tensor& scores,
    double iou_threshold) {
  PT_EAGER_TRACE;
  PT_OP_INFO("nms :", DUMP_3ARGS(boxes, scores, iou_threshold));

  // max_classes set for COCO dataset for now, can be increased in future
  // based on requirement. larger max_classes => smaller max size for
  // num_boxes allowed because of memory trade-off.
  int max_classes = 81;

  auto indices = at::zeros_like(scores, torch::kInt32);
  const int64_t box_id_out_shape{scores.sizes()[0] * max_classes};
  const int64_t shape_tensor_shape{5};

  habana::eager::EagerOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::batched_nms_eager",
      {boxes, scores, indices, iou_threshold, max_classes},
      {{box_id_out_shape}, {shape_tensor_shape}}};

  hpu_op.set_scalar_types({torch::kLong, torch::kInt});

  auto [output_nms, shape_tensor] = hpu_op.call();
  const int64_t output_numel = shape_tensor[0].item<int64_t>();

  const auto output = output_nms.slice(0, 0, output_numel, 1);

  return output;
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

at::Tensor dropout(const at::Tensor& input, double p, bool train) {
  PT_EAGER_TRACE;
  PT_OP_INFO("dropout :", DUMP_3ARGS(input, p, train));

  return std::get<0>(at::native_dropout(input, p, train));
}

// pytorch decomposes this op to at::_euclidean_dist in some cases
// For hpu we prefer to call _cdist_forward in all cases
at::Tensor cdist(
    const at::Tensor& x1,
    const at::Tensor& x2,
    const double p,
    c10::optional<int64_t> compute_mode) {
  PT_EAGER_TRACE;
  PT_OP_INFO("cdist :", DUMP_4ARGS(x1, x2, p, compute_mode));

  return _cdist_forward(x1, x2, p, compute_mode);
}

} // namespace

namespace habana::eager {

TORCH_LIBRARY(hpu, m) {
  m.def("control_edge_(Tensor(a) self)-> Tensor(a)");
  m.def(
      "hpu::cast_from_fp8(Tensor input, Tensor? scale, ScalarType out_dtype, int[]? scale_shape=None) -> Tensor");
  m.def(
      "hpu::cast_from_fp8.scalar(Tensor input, float scale, ScalarType out_dtype, int[]? scale_shape=None) -> Tensor");
  m.def(
      "hpu::cast_from_fp8.scalar_list(Tensor input, float[] scale, ScalarType out_dtype, int[]? scale_shape=None) -> Tensor");
  m.def(
      "hpu::cast_to_fp8(Tensor input, Tensor? scale, bool stochastic_rounding, Tensor(a!) out, Tensor(b!) amax) -> (Tensor(a!), Tensor(b!))");
  m.def(
      "hpu::cast_to_fp8_v2(Tensor input, Tensor? scale=None, bool stochastic_rounding=False, bool is_amax=False, ScalarType dtype=None, int[]? scale_shape=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::cast_to_fp8_v2.scalar(Tensor input, float scale, bool stochastic_rounding=False, bool is_amax=False, ScalarType dtype=None, int[]? scale_shape=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::cast_to_fp8_v2.scalar_list(Tensor input, float[] scale, bool stochastic_rounding=False, bool is_amax=False, ScalarType dtype=None, int[]? scale_shape=None) -> (Tensor, Tensor)");
  m.def(
      "hpu::convert_from_int4(Tensor input, Tensor scale, Tensor? zero_point, ScalarType out_dtype) -> Tensor");
  m.def(
      "hpu::convert_from_uint4(Tensor input, Tensor scale, Tensor? zero_point, ScalarType out_dtype) -> Tensor");
  m.def("hpu::custom_softmax(Tensor input, int flavor) -> Tensor");
  m.def(
      "hpu::fp8_gemm(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor D, ScalarType out_dtype, Tensor? A_scale_inv, Tensor? B_scale_inv, Tensor? bias, bool accumulate, Tensor(a!) out) -> Tensor(a!)");
  m.def(
      "hpu::fp8_gemm_v2(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, Tensor? A_scale_inv=None, Tensor? B_scale_inv=None, Tensor? bias=None, bool accumulate=False, int[]? B_scale_shape=None) -> Tensor");
  m.def(
      "hpu::fp8_gemm_v2.scalar(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, float A_scale_inv, float B_scale_inv, Tensor? bias=None, bool accumulate=False, int[]? B_scale_shape=None) -> Tensor");
  m.def(
      "hpu::fp8_gemm_v2.scalar_list(Tensor A, bool trans_A, Tensor B, bool trans_B, Tensor? D, ScalarType out_dtype, float[] A_scale_inv, float[] B_scale_inv, Tensor? bias=None, bool accumulate=False, int[]? B_scale_shape=None) -> Tensor");
  m.def("hpu::in_place_interleave(Tensor self) -> Tensor");
  m.def("hpu::in_place_interleave_(Tensor(a!) self) -> (Tensor(a!))");
  m.def(
      "hpu::kv_reorder(Tensor self, Tensor start, Tensor end, Tensor beam_idx) -> Tensor");
  m.def(
      "hpu::kv_reorder_(Tensor(a!) self, Tensor start, Tensor end, Tensor beam_idx) -> (Tensor(a!))");
  m.def(
      "hpu::masked_batch_gemm(Tensor a, Tensor b, Tensor mask_a, Tensor mask_b, bool trans_a, bool trans_b) -> Tensor");
  m.def(
      "hpu::optimizer_adamw(Tensor[] gradient_vec, Tensor(a!)[] weight_vec, Tensor(b!)[] exp_avg_vec, Tensor(c!)[] exp_avg_sq_vec, Tensor neg_step_t, float beta1, float beta2, float epsilon, Tensor weight_decay, bool has_weight_decay, Tensor(d!)[]? exp_avg_scales = None, Tensor(e!)[]? exp_avg_sq_scales = None) -> ()");
  m.def(
      "hpu::optimizer_ema(Tensor[] model_inputs, Tensor(a!)[] updated_ema, Tensor decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_fused_norm(Tensor[] grad, float max_norm) -> Tensor");
  m.def(
      "hpu::fused_clip_norm(Tensor(a!)[] grad, Tensor max_norm, float norm_type) -> Tensor");
  m.def(
      "hpu::optimizer_lamb_phase1(Tensor[] gradients, Tensor[] weights, Tensor(a!)[] exp_avg, Tensor(b!)[] exp_avg_sq, Tensor(c!)[] out_weight_norms, Tensor(d!)[] out_adam_norms, Tensor(e!)[] out_adam_steps, Tensor clip_global_grad_norm, int grad_averaging, float beta1, float beta2, float epsilon, Tensor bias_correction1, Tensor bias_correction2, float weight_decay) -> ()");
  m.def(
      "hpu::optimizer_lamb_phase2(Tensor(a!)[] weights, Tensor[] adam_norms, Tensor[] weight_norms, Tensor[] adam_steps, Tensor neg_step, float wd, bool use_lamb) -> ()");
  m.def(
      "hpu::optimizer_lars(Tensor[] params, Tensor(a!)[] grads, int[] skip_masks, float eeta, float weight_decay, float eps, Tensor lr) -> ()");
  m.def(
      "hpu::optimizer_resource_apply_momentum(Tensor(a!)[] params_momentum_buf_list, Tensor[] dp_list, float momentum) -> ()");
  m.def(
      "hpu::optimizer_sgd(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!) learning_rate, float wd, float mom, float damp, bool nesterov) -> ()");
  m.def(
      "hpu::optimizer_sgd_momentum(Tensor[] gradients, Tensor(a!)[] weights_in, Tensor(b!)[] momentum_in, Tensor epoch_num, Tensor(c!) learning_rate, Tensor(d!) mom, float wd, float damp, bool nesterov) -> ()");
  m.def("hpu::repeat_ht(Tensor self, Tensor result_shape) -> Tensor");
  m.def(
      "hpu::expand_ds(Tensor(a) self, Tensor shape, *, bool implicit=False) -> Tensor(a)");
  m.def(
      "hpu::ragged_softmax(Tensor self, int dim, bool half_to_float, Tensor valid_count) -> Tensor");
  m.def(
      "hpu::mixture_of_experts(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max) -> Tensor");
  m.def(
      "hpu::mixture_of_experts.fp8_measurement(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w1, Tensor[] w2, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::mixture_of_experts.fp8_measurement_fused_weights(Tensor hidden_states, Tensor expert_routing_table, Tensor router_weights, Tensor[] w12, Tensor[] w3, bool permuted_weights, str activation, int experts_min, int experts_max, bool measurement_mode) -> (Tensor, Tensor)");
  m.def(
      "hpu::rotary_pos_embedding(Tensor input, Tensor sin, Tensor cos, Tensor? position_ids, int offset, int mode) -> Tensor");
  m.def(
      "hpu::rotary_pos_embedding_backward(Tensor grad_in, Tensor sin, Tensor cos, Tensor? position_ids, int offset, int mode) -> Tensor");
  m.def(
      "hpu::ctc_loss_custom(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank, int reduction, bool zero_infinity) -> (Tensor, Tensor)");
  m.def(
      "hpu::ctc_loss_custom_backward(Tensor grad, Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, int reduction, bool zero_infinity) -> Tensor");
  m.def(
      "hpu::scaled_masked_softmax(Tensor input, Tensor mask, float scale) -> Tensor");
  m.def(
      "hpu::scaled_masked_triangular_softmax(Tensor self, Tensor start_end, float inv_scale_attn, int grouped_batch_size, bool use_max, int mode, ScalarType? out_dtype=None) -> Tensor");
  m.def(
      "hpu::scaled_triangular_softmax(Tensor self, float inv_scale_attn, Tensor? exp_sum_recpr=None, Tensor? max=None) -> Tensor");
  m.def(
      "hpu::scaled_triangular_softmax_retain(Tensor self, float inv_scale_attn) -> (Tensor, Tensor, Tensor)");
  m.def("hpu::view(Tensor input, Tensor shape) -> Tensor");
  m.def("hpu::view_neg(Tensor input, Tensor shape, int[] shape) -> Tensor");
  m.def("hpu::slice_ht(Tensor input, Tensor shape, Tensor shape) -> Tensor");
  m.def(
      "hpu::slice_ds(Tensor input, SymInt dim, SymInt start, SymInt end, SymInt step, SymInt[]? size=None) -> Tensor");
  m.def(
      "strided_insert_orig_ds(Tensor self, Tensor other, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "strided_insert_orig_ds_h2d(Tensor self, Tensor other, Tensor stride) -> (Tensor)");
  m.def(
      "strided_view_ds_h2d(Tensor self, Tensor size, Tensor stride, Tensor offset) -> (Tensor)");
  m.def(
      "hpu::select_scatter(Tensor self, Tensor src, Tensor dim, Tensor index) -> (Tensor)");
  m.def(
      "hpu::slice_scatter(Tensor self, Tensor src, Tensor dim = None, Tensor? start = None, Tensor? end = None, Tensor step = None) -> (Tensor)");
  m.def(
      "hpu::slice_scatter_ds(Tensor self, Tensor src, Tensor step = None, Tensor start = None) -> (Tensor)");
  m.def(
      "hpu::as_strided_scatter(Tensor self, Tensor src, Tensor stride, Tensor? storage_offset = None) -> (Tensor)");
  m.def(
      "hpu::as_strided_scatter_orig(Tensor self, Tensor src, Tensor stride) -> (Tensor)");
  m.def(
      "strided_view_orig_ds_h2d(Tensor self, Tensor size, Tensor stride) -> (Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  // for torch.compile to insert seed
  m.def(
      "hpu::sdpa_recomp_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_modem, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_recomp_bwd(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, Tensor m, Tensor linv, Tensor ? seed, bool is_causal, float p, float scale, str softmax_mode, Tensor fwd_out) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::sdpa_bwd(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor P, Tensor? dm, bool is_causal, float p, float scale, Tensor fwd_out) -> (Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, Tensor? valid_seq_len, str seq_padding_type) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_bwd(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor P, Tensor? dm, bool is_causal, float p, float scale, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? d_scale_s, Tensor? d_scale_do, Tensor? d_scale_ds, Tensor? q_scale_s, Tensor? q_scale_ds, bool is_amax_ds, Tensor fwd_out) -> (Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout_seed(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, Tensor? d_scale_q, Tensor? d_scale_k, Tensor? d_scale_v, Tensor? q_scale_s, Tensor? q_scale_o, Tensor? d_scale_s, bool is_amax_s, bool is_amax_o, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0, Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout.scalar(Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_0,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");
  m.def(
      "hpu::fp8_sdpa_recomp_fwd_dropout_seed.scalar(Tensor seed, Tensor q, Tensor k, Tensor v, Tensor? attention_mask, float p, float scale, bool is_causal, bool requires_backward, str softmax_mode, float d_scale_q, float d_scale_k, float d_scale_v, float q_scale_s, float q_scale_o, float d_scale_s, bool is_amax_s, bool is_amax_o,  Tensor? valid_seq_len, str seq_padding_type ) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def("hpu::accumulate_grads_(Tensor[] variables, Tensor[] new_grads) -> ()");
  m.def("hpu::custom_foreach_add_(Tensor(a!)[] self, Tensor[] other) -> ()");
  m.def(
      "hpu::batched_nms_eager(Tensor boxes, Tensor scores, Tensor indexes, float iou_threshold, int max_classes) -> (Tensor, Tensor)");
  m.def(
      "hpu::habana_randperm_ht(Tensor seed, Tensor h2d_tensor, Tensor shape_tensor, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_rand_st(Tensor seed, Tensor shape_tensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_randn_st(Tensor seed, Tensor shape_tensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_randint_st(Tensor seed, SymInt low, SymInt high, Tensor shape_tensor, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::full_ds(Tensor size, Scalar fill_value, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::empty_ds(Tensor size,  ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor");
  m.def(
      "hpu::constant_pad_nd(Tensor input, Tensor pad_tensor, Tensor output_shape_tensor, Scalar value) -> Tensor");
  m.def(
      "hpu::constant_pad_nd_ds(Tensor input, SymInt[] pad, Scalar value, SymInt[]? size=None) -> Tensor");
  m.def("hpu::weight_permutation(Tensor input) -> Tensor");
  m.def(
      "hpu::custom_bernoulli.Size(SymInt[] size, float p, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor");
  m.def(
      "hpu::habana_seed_generator(Tensor seed, Tensor counter, int size) -> Tensor");
  HABANA_RANDOM_DEF(bernoulli, "Tensor seed, Tensor self")
  HABANA_RANDOM_DEF_VARIANT(bernoulli, p, "Tensor seed, Tensor self, float p")
  HABANA_RANDOM_DEF_VARIANT(
      bernoulli, Tensor, "Tensor seed, Tensor self, Tensor p")
  HABANA_RANDOM_DEF_VARIANT(
      bernoulli,
      Size,
      "Tensor seed, SymInt[] size, Scalar p, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF(poisson, "Tensor seed, Tensor self")
  HABANA_RANDOM_DEF(
      rand,
      "Tensor seed, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF(
      randn,
      "Tensor seed, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF(
      randint,
      "Tensor seed, SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF(
      multinomial,
      "Tensor seed, Tensor self, int num_samples, bool replacement=False")
  HABANA_RANDOM_DEF(
      uniform, "Tensor seed, Tensor self, float from=0, float to=1")
  HABANA_RANDOM_DEF(
      randperm,
      "Tensor seed, SymInt n, *, ScalarType? dtype=long, Layout? layout=None, Device? device=None, bool? pin_memory=None")
  HABANA_RANDOM_DEF_2_OUTS(
      native_dropout, "Tensor seed, Tensor input, float p, bool? train")
}

TORCH_LIBRARY_IMPL(hpu, HPU, m) {
  m.impl("hpu::accumulate_grads_", accumulate_grads_);
  m.impl("hpu::cast_from_fp8", cast_from_fp8);
  m.impl("hpu::cast_from_fp8.scalar", cast_from_fp8_scalar);
  m.impl("hpu::cast_from_fp8.scalar_list", cast_from_fp8_scalar_list);
  m.impl("hpu::cast_to_fp8", cast_to_fp8);
  m.impl("hpu::cast_to_fp8_v2", cast_to_fp8_v2);
  m.impl("hpu::cast_to_fp8_v2.scalar", cast_to_fp8_v2_scalar);
  m.impl("hpu::cast_to_fp8_v2.scalar_list", cast_to_fp8_v2_scalar_list);
  m.impl("hpu::convert_from_int4", convert_from_int4);
  m.impl("hpu::convert_from_uint4", convert_from_uint4);
  m.impl("hpu::custom_softmax", custom_softmax);
  m.impl("hpu::fp8_gemm", fp8_gemm);
  m.impl("hpu::fp8_gemm_v2", fp8_gemm_v2);
  m.impl("hpu::fp8_gemm_v2.scalar", fp8_gemm_v2_scalar);
  m.impl("hpu::fp8_gemm_v2.scalar_list", fp8_gemm_v2_scalar_list);
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
  m.impl("hpu::ragged_softmax", _ragged_softmax);
  m.impl("hpu::mixture_of_experts", mixture_of_experts);
  m.impl(
      "hpu::mixture_of_experts.fused_weights",
      mixture_of_experts_fused_weights);
  m.impl(
      "hpu::mixture_of_experts.fp8_measurement",
      mixture_of_experts_fp8_measurement);
  m.impl(
      "hpu::mixture_of_experts.fp8_measurement_fused_weights",
      mixture_of_experts_fp8_measurement_fused_weights);
  m.impl("hpu::optimizer_sgd", optimizer_sgd);
  m.impl("hpu::optimizer_sgd_momentum", optimizer_sgd_momentum);
  m.impl("hpu::rotary_pos_embedding", rotary_pos_embedding);
  m.impl("hpu::rotary_pos_embedding_backward", rotary_pos_embedding_backward);
  m.impl("hpu::ctc_loss_custom", ctc_loss_custom);
  m.impl("hpu::ctc_loss_custom_backward", ctc_loss_custom_backward);
  m.impl("hpu::scaled_masked_softmax", scaled_masked_softmax);
  m.impl(
      "hpu::scaled_masked_triangular_softmax",
      scaled_masked_triangular_softmax);
  m.impl("hpu::scaled_triangular_softmax", scaled_triangular_softmax);
  m.impl("hpu::sdpa_recomp_fwd", sdpa_recomp_fwd);
  m.impl("hpu::sdpa_recomp_fwd_non_dropout", sdpa_recomp_fwd);
  m.impl("hpu::sdpa_recomp_bwd", sdpa_recomp_bwd);
  m.impl("hpu::sdpa_fwd", sdpa_fwd);
  m.impl("hpu::sdpa_fwd_non_dropout", sdpa_fwd);
  m.impl("hpu::sdpa_fwd_dropout", sdpa_fwd);
  m.impl("hpu::fp8_sdpa_fwd", fp8_sdpa_fwd);
  m.impl("hpu::fp8_sdpa_fwd_non_dropout", fp8_sdpa_fwd);
  m.impl("hpu::fp8_sdpa_fwd_dropout", fp8_sdpa_fwd);
  m.impl("hpu::fp8_sdpa_bwd", fp8_sdpa_bwd);
  m.impl("hpu::sdpa_bwd", sdpa_bwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd", fp8_sdpa_recomp_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd_non_dropout", fp8_sdpa_recomp_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd_dropout", fp8_sdpa_recomp_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd.scalar", fp8_sdpa_recomp_scalar_fwd);
  m.impl(
      "hpu::fp8_sdpa_recomp_fwd_non_dropout.scalar",
      fp8_sdpa_recomp_scalar_fwd);
  m.impl("hpu::fp8_sdpa_recomp_fwd_dropout.scalar", fp8_sdpa_recomp_scalar_fwd);
  m.impl(
      "hpu::scaled_triangular_softmax_retain",
      scaled_triangular_softmax_retain);
  m.impl("hpu::slice_ds", slice_ds);
  m.impl("hpu::constant_pad_nd_ds", constant_pad_nd_ds);
  m.impl("hpu::fused_clip_norm", fused_clip_norm);
  m.impl("hpu::weight_permutation", weight_permutation);
}

TORCH_LIBRARY_IMPL(aten, HPU, m) {
  m.impl("cdist", cdist);
  m.impl("dropout", dropout);
}

TORCH_LIBRARY_IMPL(torchvision, HPU, m) {
  m.impl("roi_align", roi_align);
  m.impl("_roi_align_backward", roi_align_backward);
  m.impl("nms", nms);
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
