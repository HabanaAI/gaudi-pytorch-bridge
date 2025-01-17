/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "hpu_ops/mixture_of_experts.h"
#include <habana_kernels/mixture_of_experts.h>
#include "common/dump_args.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_lazy/hpu_stage_submission.h"
#include "habana_lazy/lazy_executor.h"
#include "hpu_ops/op_logger.h"

using namespace habana;

namespace habana_lazy {

at::Tensor mixture_of_experts_lazy(
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
  PT_LAZY_OP_TRACE;
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

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
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

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fwd_lazy(
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
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd :",
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

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fwd_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd.fused_weights :",
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

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts_fwd",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_bwd_lazy(
    const at::Tensor& grad,
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
  PT_LAZY_OP_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd :",
      DUMP_11ARGS(
          grad,
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

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts_bwd",
      {grad,
       hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_bwd_fused_weights_lazy(
    const at::Tensor& grad,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_bwd.fused_weights :",
      DUMP_10ARGS(
          grad,
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts_bwd",
      {grad,
       hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement_lazy(
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
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

  LazyOp<std::tuple<at::Tensor, at::Tensor>> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       measurement_mode},
      {hidden_states.sizes().vec(), {static_cast<int64_t>(w1.size())}},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

std::tuple<at::Tensor, at::Tensor>
mixture_of_experts_fp8_measurement_fused_weights_lazy(
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
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

  LazyOp<std::tuple<at::Tensor, at::Tensor>> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max,
       measurement_mode},
      {hidden_states.sizes().vec(), {static_cast<int64_t>(w12.size())}},
      0};

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fp8_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8 :",
      DUMP_15ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w1,
       d_scale_w2,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};
  op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fp8_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights :",
      DUMP_13ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w12,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};
  op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fp8_scalars_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_intermediate_hidden_states,
    const c10::ArrayRef<double>& d_scale_w1,
    const c10::ArrayRef<double>& d_scale_w2,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_scalars :",
      DUMP_15ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w1,
          w2,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w1,
          d_scale_w2,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w1,
       d_scale_w2,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};
  op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

at::Tensor mixture_of_experts_fp8_fused_weights_scalars_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_intermediate_hidden_states,
    const c10::ArrayRef<double>& d_scale_w12,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fp8_fused_weights_scalars :",
      DUMP_13ARGS(
          hidden_states,
          expert_routing_table,
          router_weights,
          w12,
          w3,
          d_scale_hidden_states,
          d_scale_intermediate_hidden_states,
          d_scale_w12,
          d_scale_w3,
          permuted_weights,
          activation,
          experts_min,
          experts_max));

  LazyOp<at::Tensor> op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max},
      {hidden_states.sizes().vec()},
      0};
  op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);

  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, op)
}

class MixtureOfExpertsFunction
    : public torch::autograd::Function<MixtureOfExpertsFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& hidden_states,
      const torch::autograd::Variable& expert_routing_table,
      const torch::autograd::Variable& router_weights,
      const c10::ArrayRef<torch::autograd::Variable>& w1,
      const c10::ArrayRef<torch::autograd::Variable>& w2,
      const c10::ArrayRef<torch::autograd::Variable>& w3,
      const bool permuted_weights,
      const c10::string_view activation,
      const int64_t experts_min,
      const int64_t experts_max) {
    at::AutoDispatchBelowADInplaceOrView g;

    int64_t num_experts = w1.size();
    torch::autograd::variable_list to_save;
    to_save.reserve(non_list_tensors + weights_list_amount * num_experts);
    to_save.push_back(hidden_states);
    to_save.push_back(expert_routing_table);
    to_save.push_back(router_weights);
    to_save.insert(to_save.begin() + non_list_tensors, w1.begin(), w1.end());
    to_save.insert(
        to_save.begin() + non_list_tensors + num_experts, w2.begin(), w2.end());
    to_save.insert(
        to_save.begin() + non_list_tensors + 2 * num_experts,
        w3.begin(),
        w3.end());

    ctx->save_for_backward(to_save);
    ctx->saved_data["num_experts"] = num_experts;
    ctx->saved_data["permuted_weights"] = permuted_weights;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["experts_min"] = experts_min;
    ctx->saved_data["experts_max"] = experts_max;

    auto output = mixture_of_experts_fwd_lazy(
        hidden_states,
        expert_routing_table,
        router_weights,
        w1,
        w2,
        w3,
        permuted_weights,
        activation,
        experts_min,
        experts_max);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    auto num_experts = ctx->saved_data["num_experts"].toInt();
    auto w1 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors,
        saved_vars.begin() + non_list_tensors + num_experts);
    auto w2 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors + num_experts,
        saved_vars.begin() + non_list_tensors + 2 * num_experts);
    auto w3 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors + 2 * num_experts,
        saved_vars.begin() + non_list_tensors +
            weights_list_amount * num_experts);

    auto result = mixture_of_experts_bwd_lazy(
        grads[0],
        saved_vars[0],
        saved_vars[1],
        saved_vars[2],
        w1,
        w2,
        w3,
        ctx->saved_data["permuted_weights"].toBool(),
        ctx->saved_data["activation"].toStringRef(),
        ctx->saved_data["experts_min"].toInt(),
        ctx->saved_data["experts_max"].toInt());

    torch::autograd::variable_list grad_input;
    grad_input.reserve(non_list_inputs + weights_list_amount * num_experts);
    for (int64_t i = 0; i < non_list_inputs + weights_list_amount * num_experts;
         i++) {
      grad_input.push_back(at::Tensor());
    }
    grad_input[0] = result;
    return grad_input;
  }

 private:
  static const int64_t weights_list_amount = 3;
  static const int64_t non_list_tensors = 3;
  static const int64_t non_list_inputs = 7;
};

class MixtureOfExpertsFusedWeightsFunction
    : public torch::autograd::Function<MixtureOfExpertsFusedWeightsFunction> {
 public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& hidden_states,
      const torch::autograd::Variable& expert_routing_table,
      const torch::autograd::Variable& router_weights,
      const c10::ArrayRef<torch::autograd::Variable>& w12,
      const c10::ArrayRef<torch::autograd::Variable>& w3,
      const bool permuted_weights,
      const c10::string_view activation,
      const int64_t experts_min,
      const int64_t experts_max) {
    at::AutoDispatchBelowADInplaceOrView g;

    int64_t num_experts = w12.size();
    torch::autograd::variable_list to_save;
    to_save.reserve(non_list_tensors + weights_list_amount * num_experts);
    to_save.push_back(hidden_states);
    to_save.push_back(expert_routing_table);
    to_save.push_back(router_weights);
    to_save.insert(to_save.begin() + non_list_tensors, w12.begin(), w12.end());
    to_save.insert(
        to_save.begin() + non_list_tensors + num_experts, w3.begin(), w3.end());
    ctx->save_for_backward(to_save);

    ctx->saved_data["num_experts"] = num_experts;
    ctx->saved_data["permuted_weights"] = permuted_weights;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["experts_min"] = experts_min;
    ctx->saved_data["experts_max"] = experts_max;

    auto output = mixture_of_experts_fwd_fused_weights_lazy(
        hidden_states,
        expert_routing_table,
        router_weights,
        w12,
        w3,
        permuted_weights,
        activation,
        experts_min,
        experts_max);

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();

    int64_t num_experts = ctx->saved_data["num_experts"].toInt();
    auto w12 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors,
        saved_vars.begin() + non_list_tensors + num_experts);
    auto w3 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_tensors + num_experts,
        saved_vars.begin() + non_list_tensors + 2 * num_experts);

    auto result = mixture_of_experts_bwd_fused_weights_lazy(
        grads[0],
        saved_vars[0],
        saved_vars[1],
        saved_vars[2],
        w12,
        w3,
        ctx->saved_data["permuted_weights"].toBool(),
        ctx->saved_data["activation"].toStringRef(),
        ctx->saved_data["experts_min"].toInt(),
        ctx->saved_data["experts_max"].toInt());

    torch::autograd::variable_list grad_input;
    grad_input.reserve(non_list_inputs + weights_list_amount * num_experts);
    for (int64_t i = 0; i < non_list_inputs + weights_list_amount * num_experts;
         i++) {
      grad_input.push_back(at::Tensor());
    }
    grad_input[0] = result;
    return grad_input;
  }

 private:
  static const int64_t weights_list_amount = 2;
  static const int64_t non_list_tensors = 3;
  static const int64_t non_list_inputs = 7;
};

at::Tensor mixture_of_experts_fwd_autograd_lazy(
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
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts.fwd :",
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

  return MixtureOfExpertsFunction::apply(
      hidden_states,
      expert_routing_table,
      router_weights,
      w1,
      w2,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max)[0];
}

at::Tensor mixture_of_experts_fwd_fused_weights_autograd_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max) {
  PT_LAZY_OP_TRACE;
  PT_LAZY_TRACE;
  PT_OP_INFO(
      "mixture_of_experts_fwd_fused_weights :",
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

  return MixtureOfExpertsFusedWeightsFunction::apply(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max)[0];
}

} // namespace habana_lazy
