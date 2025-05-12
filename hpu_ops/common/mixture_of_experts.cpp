/**
 * Copyright (c) 2025 Intel Corporation
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

#include "hpu_ops/common/mixture_of_experts.h"
#include <ATen/ATen.h>
#include "common/dump_args.h"
#include "common/utils.h"
#include "generated/autograd/autograd_ops.h"
#include "habana_helpers/logging.h"
#include "hpu_ops/op_logger.h"

namespace habana {

std::vector<Tensor> MixtureOfExpertsFwdFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    at::TensorList w1,
    at::TensorList w2,
    at::TensorList w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max) {
  at::AutoDispatchBelowADInplaceOrView g;

  size_t num_experts = w1.size();
  torch::autograd::variable_list to_save;
  to_save.reserve(weights_per_expert * num_experts + outputs_for_bwd);
  to_save.insert(to_save.begin(), w1.begin(), w1.end());
  to_save.insert(to_save.begin() + num_experts, w2.begin(), w2.end());
  to_save.insert(to_save.begin() + 2 * num_experts, w3.begin(), w3.end());

  ctx->saved_data["num_experts"] = static_cast<int64_t>(num_experts);
  ctx->saved_data["permuted_weights"] = permuted_weights;
  ctx->saved_data["activation"] = activation;
  ctx->saved_data["experts_min"] = experts_min;
  ctx->saved_data["experts_max"] = experts_max;
  ctx->saved_data["router_weights_size"] = router_weights.sizes();

  auto outputs = mixture_of_experts_fwd(
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

  for (size_t i = 1; i <= outputs_for_bwd; i++) {
    // First output is not needed for backward
    to_save.push_back(outputs[i]);
  }
  ctx->save_for_backward(to_save);

  return outputs;
}

std::vector<Tensor> MixtureOfExpertsFwdFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
  size_t num_experts = ctx->saved_data["num_experts"].toInt();

  auto w1 = torch::autograd::variable_list(
      saved_vars.begin(), saved_vars.begin() + num_experts);
  auto w2 = torch::autograd::variable_list(
      saved_vars.begin() + num_experts, saved_vars.begin() + 2 * num_experts);
  auto w3 = torch::autograd::variable_list(
      saved_vars.begin() + 2 * num_experts,
      saved_vars.begin() + 3 * num_experts);

  const size_t first_intermediate_fwd_output = weights_per_expert * num_experts;
  const auto& chunks_input = saved_vars[first_intermediate_fwd_output];
  const auto& token_to_chunk = saved_vars[first_intermediate_fwd_output + 1];
  const auto& token_in_chunk = saved_vars[first_intermediate_fwd_output + 2];
  const auto& chunks_routing_table =
      saved_vars[first_intermediate_fwd_output + 3];
  const auto& chunks_router_weights =
      saved_vars[first_intermediate_fwd_output + 4];
  const auto& gemm1_out = saved_vars[first_intermediate_fwd_output + 5];
  const auto& gemm2_out = saved_vars[first_intermediate_fwd_output + 6];
  const auto& activation_out = saved_vars[first_intermediate_fwd_output + 7];
  const auto& mult_out = saved_vars[first_intermediate_fwd_output + 8];
  const auto& mlp_out = saved_vars[first_intermediate_fwd_output + 9];

  auto result = mixture_of_experts_bwd(
      grads[0],
      chunks_input,
      token_to_chunk,
      token_in_chunk,
      chunks_routing_table,
      chunks_router_weights,
      gemm1_out,
      gemm2_out,
      activation_out,
      mult_out,
      mlp_out,
      w1,
      w2,
      w3,
      ctx->saved_data["permuted_weights"].toBool(),
      ctx->saved_data["activation"].toStringRef(),
      ctx->saved_data["experts_min"].toInt(),
      ctx->saved_data["experts_max"].toInt(),
      ctx->saved_data["router_weights_size"].toIntVector());

  const size_t num_fwd_inputs =
      non_list_inputs + weights_per_expert * num_experts;
  torch::autograd::variable_list grad_input(num_fwd_inputs, at::Tensor());

  grad_input[0] = result[0];
  grad_input[2] = result[1];
  for (size_t i = 0; i < num_experts; i++) {
    grad_input[non_list_tensors + i] = result[2 + i];
    grad_input[non_list_tensors + num_experts + i] =
        result[2 + num_experts + i];
    grad_input[non_list_tensors + 2 * num_experts + i] =
        result[2 + 2 * num_experts + i];
  }
  return grad_input;
}

at::Tensor MixtureOfExpertsRecompFwdFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const Tensor& hidden_states,
    const Tensor& expert_routing_table,
    const Tensor& router_weights,
    TensorList w1,
    TensorList w2,
    TensorList w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max) {
  at::AutoDispatchBelowADInplaceOrView g;

  size_t num_experts = w1.size();
  torch::autograd::variable_list to_save;
  to_save.reserve(non_list_tensors + weights_per_expert * num_experts);
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

  ctx->saved_data["num_experts"] = static_cast<int64_t>(num_experts);
  ctx->saved_data["permuted_weights"] = permuted_weights;
  ctx->saved_data["activation"] = activation;
  ctx->saved_data["experts_min"] = experts_min;
  ctx->saved_data["experts_max"] = experts_max;
  ctx->save_for_backward(to_save);

  auto output = mixture_of_experts_recomp_fwd(
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

  return output;
}

std::vector<at::Tensor> MixtureOfExpertsRecompFwdFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
  size_t num_experts = ctx->saved_data["num_experts"].toInt();

  auto w1 = torch::autograd::variable_list(
      saved_vars.begin() + non_list_tensors,
      saved_vars.begin() + non_list_tensors + num_experts);
  auto w2 = torch::autograd::variable_list(
      saved_vars.begin() + non_list_tensors + num_experts,
      saved_vars.begin() + non_list_tensors + 2 * num_experts);
  auto w3 = torch::autograd::variable_list(
      saved_vars.begin() + non_list_tensors + 2 * num_experts,
      saved_vars.begin() + non_list_tensors + 3 * num_experts);

  auto result = mixture_of_experts_recomp_bwd(
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

  const size_t num_fwd_inputs =
      non_list_inputs + weights_per_expert * num_experts;
  torch::autograd::variable_list grad_input(num_fwd_inputs, at::Tensor());

  grad_input[0] = result[0];
  grad_input[2] = result[1];
  for (size_t i = 0; i < num_experts; i++) {
    grad_input[non_list_tensors + i] = result[2 + i];
    grad_input[non_list_tensors + num_experts + i] =
        result[2 + num_experts + i];
    grad_input[non_list_tensors + 2 * num_experts + i] =
        result[2 + 2 * num_experts + i];
  }
  return grad_input;
}

torch::autograd::variable_list MixtureOfExpertsFwdFusedWeightsFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const Tensor& hidden_states,
    const Tensor& expert_routing_table,
    const Tensor& router_weights,
    TensorList w12,
    TensorList w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max) {
  at::AutoDispatchBelowADInplaceOrView g;

  int64_t num_experts = w12.size();
  torch::autograd::variable_list to_save;
  to_save.reserve(
      weights_per_expert_fused * num_experts + outputs_for_bwd_fused);
  to_save.insert(to_save.begin(), w12.begin(), w12.end());
  to_save.insert(to_save.begin() + num_experts, w3.begin(), w3.end());

  ctx->saved_data["num_experts"] = static_cast<int64_t>(num_experts);
  ctx->saved_data["permuted_weights"] = permuted_weights;
  ctx->saved_data["activation"] = activation;
  ctx->saved_data["experts_min"] = experts_min;
  ctx->saved_data["experts_max"] = experts_max;
  ctx->saved_data["router_weights_size"] = router_weights.sizes();

  auto outputs = mixture_of_experts_fwd_fused_weights(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);

  for (size_t i = 1; i <= outputs_for_bwd_fused; i++) {
    to_save.push_back(outputs[i]);
  }
  ctx->save_for_backward(to_save);

  return outputs;
}

std::vector<at::Tensor> MixtureOfExpertsFwdFusedWeightsFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
  size_t num_experts = ctx->saved_data["num_experts"].toInt();

  auto w12 = torch::autograd::variable_list(
      saved_vars.begin(), saved_vars.begin() + num_experts);
  auto w3 = torch::autograd::variable_list(
      saved_vars.begin() + num_experts, saved_vars.begin() + 2 * num_experts);

  const size_t first_intermediate_fwd_output =
      weights_per_expert_fused * num_experts;

  const auto& chunks_input = saved_vars[first_intermediate_fwd_output];
  const auto& token_to_chunk = saved_vars[first_intermediate_fwd_output + 1];
  const auto& token_in_chunk = saved_vars[first_intermediate_fwd_output + 2];
  const auto& chunks_routing_table =
      saved_vars[first_intermediate_fwd_output + 3];
  const auto& chunks_router_weights =
      saved_vars[first_intermediate_fwd_output + 4];
  const auto& gemm12_out = saved_vars[first_intermediate_fwd_output + 5];
  const auto& activation_out = saved_vars[first_intermediate_fwd_output + 6];
  const auto& mult_out = saved_vars[first_intermediate_fwd_output + 7];
  const auto& mlp_out = saved_vars[first_intermediate_fwd_output + 8];

  auto result = mixture_of_experts_bwd_fused_weights(
      grads[0],
      chunks_input,
      token_to_chunk,
      token_in_chunk,
      chunks_routing_table,
      chunks_router_weights,
      gemm12_out,
      activation_out,
      mult_out,
      mlp_out,
      w12,
      w3,
      ctx->saved_data["permuted_weights"].toBool(),
      ctx->saved_data["activation"].toStringRef(),
      ctx->saved_data["experts_min"].toInt(),
      ctx->saved_data["experts_max"].toInt(),
      ctx->saved_data["router_weights_size"].toIntVector());

  const size_t num_fwd_inputs =
      non_list_inputs + weights_per_expert_fused * num_experts;
  torch::autograd::variable_list grad_input(num_fwd_inputs, at::Tensor());

  grad_input[0] = result[0];
  grad_input[2] = result[1];
  for (size_t i = 0; i < num_experts; i++) {
    grad_input[non_list_tensors + i] = result[2 + i];
    grad_input[non_list_tensors + num_experts + i] =
        result[2 + num_experts + i];
  }
  return grad_input;
}

at::Tensor MixtureOfExpertsRecompFwdFusedWeightsFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const Tensor& hidden_states,
    const Tensor& expert_routing_table,
    const Tensor& router_weights,
    TensorList w12,
    TensorList w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max) {
  at::AutoDispatchBelowADInplaceOrView g;

  size_t num_experts = w12.size();
  torch::autograd::variable_list to_save;
  to_save.reserve(non_list_tensors + weights_per_expert_fused * num_experts);
  to_save.push_back(hidden_states);
  to_save.push_back(expert_routing_table);
  to_save.push_back(router_weights);
  to_save.insert(to_save.begin() + non_list_tensors, w12.begin(), w12.end());
  to_save.insert(
      to_save.begin() + non_list_tensors + num_experts, w3.begin(), w3.end());

  ctx->saved_data["num_experts"] = static_cast<int64_t>(num_experts);
  ctx->saved_data["permuted_weights"] = permuted_weights;
  ctx->saved_data["activation"] = activation;
  ctx->saved_data["experts_min"] = experts_min;
  ctx->saved_data["experts_max"] = experts_max;
  ctx->save_for_backward(to_save);
  auto output = mixture_of_experts_recomp_fwd_fused_weights(
      hidden_states,
      expert_routing_table,
      router_weights,
      w12,
      w3,
      permuted_weights,
      activation,
      experts_min,
      experts_max);

  return output;
}

std::vector<at::Tensor> MixtureOfExpertsRecompFwdFusedWeightsFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  torch::autograd::variable_list saved_vars = ctx->get_saved_variables();

  size_t num_experts = ctx->saved_data["num_experts"].toInt();
  auto w12 = torch::autograd::variable_list(
      saved_vars.begin() + non_list_tensors,
      saved_vars.begin() + non_list_tensors + num_experts);
  auto w3 = torch::autograd::variable_list(
      saved_vars.begin() + non_list_tensors + num_experts,
      saved_vars.begin() + non_list_tensors + 2 * num_experts);

  auto result = mixture_of_experts_recomp_bwd_fused_weights(
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

  const size_t num_fwd_inputs =
      non_list_inputs + weights_per_expert_fused * num_experts;
  torch::autograd::variable_list grad_input(num_fwd_inputs, at::Tensor());

  grad_input[0] = result[0];
  grad_input[2] = result[1];
  for (size_t i = 0; i < num_experts; i++) {
    grad_input[non_list_tensors + i] = result[2 + i];
    grad_input[non_list_tensors + num_experts + i] =
        result[2 + num_experts + i];
  }
  return grad_input;
}

at::Tensor mixture_of_experts_fwd_autograd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp) {
  PT_OP_INFO(
      "mixture_of_experts.fwd :",
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
          recomp));
  return recomp.value_or(true) ? MixtureOfExpertsRecompFwdFunction::apply(
                                     hidden_states,
                                     expert_routing_table,
                                     router_weights,
                                     w1,
                                     w2,
                                     w3,
                                     permuted_weights,
                                     activation,
                                     experts_min,
                                     experts_max)
                               : MixtureOfExpertsFwdFunction::apply(
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

at::Tensor mixture_of_experts_fwd_fused_weights_autograd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp) {
  PT_OP_INFO(
      "mixture_of_experts_fwd_fused_weights :",
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
          recomp));

  return recomp.value_or(true)
      ? MixtureOfExpertsRecompFwdFusedWeightsFunction::apply(
            hidden_states,
            expert_routing_table,
            router_weights,
            w12,
            w3,
            permuted_weights,
            activation,
            experts_min,
            experts_max)
      : MixtureOfExpertsFwdFusedWeightsFunction::apply(
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

} // namespace habana
