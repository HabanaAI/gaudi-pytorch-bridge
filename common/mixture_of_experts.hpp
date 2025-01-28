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
#pragma once

#include <ATen/ATen.h>

namespace habana {

std::vector<at::Tensor> mixture_of_experts_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

at::Tensor mixture_of_experts_recomp_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_fwd_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

at::Tensor mixture_of_experts_recomp_fwd_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& router_weights,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& gemm1_out,
    const at::Tensor& gemm2_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_recomp_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_bwd_fused_weights(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& router_weights,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& gemm12_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_recomp_bwd_fused_weights(
    const at::Tensor& grad,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

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

    size_t num_experts = w1.size();
    torch::autograd::variable_list to_save;
    to_save.reserve(
        non_list_saved_tensors + weights_per_expert * num_experts +
        outputs_for_bwd);
    to_save.push_back(router_weights);
    to_save.insert(
        to_save.begin() + non_list_saved_tensors, w1.begin(), w1.end());
    to_save.insert(
        to_save.begin() + non_list_saved_tensors + num_experts,
        w2.begin(),
        w2.end());
    to_save.insert(
        to_save.begin() + non_list_saved_tensors + 2 * num_experts,
        w3.begin(),
        w3.end());

    ctx->saved_data["num_experts"] = static_cast<int64_t>(num_experts);
    ctx->saved_data["permuted_weights"] = permuted_weights;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["experts_min"] = experts_min;
    ctx->saved_data["experts_max"] = experts_max;

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

    return {outputs[0]};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    size_t num_experts = ctx->saved_data["num_experts"].toInt();

    auto w1 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_saved_tensors,
        saved_vars.begin() + non_list_saved_tensors + num_experts);
    auto w2 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_saved_tensors + num_experts,
        saved_vars.begin() + non_list_saved_tensors + 2 * num_experts);
    auto w3 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_saved_tensors + 2 * num_experts,
        saved_vars.begin() + non_list_saved_tensors + 3 * num_experts);

    const size_t first_intermediate_fwd_output =
        non_list_saved_tensors + weights_per_expert * num_experts;
    const auto& chunks_input = saved_vars[first_intermediate_fwd_output];
    const auto& token_to_chunk = saved_vars[first_intermediate_fwd_output + 1];
    const auto& token_in_chunk = saved_vars[first_intermediate_fwd_output + 2];
    const auto& chunks_routing_table =
        saved_vars[first_intermediate_fwd_output + 3];
    const auto& gemm1_out = saved_vars[first_intermediate_fwd_output + 4];
    const auto& gemm2_out = saved_vars[first_intermediate_fwd_output + 5];
    const auto& activation_out = saved_vars[first_intermediate_fwd_output + 6];
    const auto& mult_out = saved_vars[first_intermediate_fwd_output + 7];

    auto result = mixture_of_experts_bwd(
        grads[0],
        saved_vars[0],
        chunks_input,
        token_to_chunk,
        token_in_chunk,
        chunks_routing_table,
        gemm1_out,
        gemm2_out,
        activation_out,
        mult_out,
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
    for (size_t i = 0; i < num_experts; i++) {
      grad_input[non_list_tensors + i] = result[1 + i];
      grad_input[non_list_tensors + num_experts + i] =
          result[1 + num_experts + i];
      grad_input[non_list_tensors + 2 * num_experts + i] =
          result[1 + 2 * num_experts + i];
    }
    return grad_input;
  }

 private:
  static const size_t weights_per_expert = 3;
  static const size_t non_list_tensors = 3;
  static const size_t non_list_saved_tensors = 1;
  static const size_t non_list_inputs = 7;
  static const size_t outputs_for_bwd = 8;
};

class MixtureOfExpertsRecompFunction
    : public torch::autograd::Function<MixtureOfExpertsRecompFunction> {
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

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
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
    for (size_t i = 0; i < num_experts; i++) {
      grad_input[non_list_tensors + i] = result[1 + i];
      grad_input[non_list_tensors + num_experts + i] =
          result[1 + num_experts + i];
      grad_input[non_list_tensors + 2 * num_experts + i] =
          result[1 + 2 * num_experts + i];
    }
    return grad_input;
  }

 private:
  static const size_t weights_per_expert = 3;
  static const size_t non_list_tensors = 3;
  static const size_t non_list_inputs = 7;
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
    to_save.reserve(
        non_list_saved_tensors + weights_per_expert * num_experts +
        outputs_for_bwd);
    to_save.push_back(router_weights);
    to_save.insert(
        to_save.begin() + non_list_saved_tensors, w12.begin(), w12.end());
    to_save.insert(
        to_save.begin() + non_list_saved_tensors + num_experts,
        w3.begin(),
        w3.end());

    ctx->saved_data["num_experts"] = static_cast<int64_t>(num_experts);
    ctx->saved_data["permuted_weights"] = permuted_weights;
    ctx->saved_data["activation"] = activation;
    ctx->saved_data["experts_min"] = experts_min;
    ctx->saved_data["experts_max"] = experts_max;

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

    for (size_t i = 1; i <= outputs_for_bwd; i++) {
      to_save.push_back(outputs[i]);
    }
    ctx->save_for_backward(to_save);

    return {outputs[0]};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
    torch::autograd::variable_list saved_vars = ctx->get_saved_variables();
    size_t num_experts = ctx->saved_data["num_experts"].toInt();

    auto w12 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_saved_tensors,
        saved_vars.begin() + non_list_saved_tensors + num_experts);
    auto w3 = torch::autograd::variable_list(
        saved_vars.begin() + non_list_saved_tensors + num_experts,
        saved_vars.begin() + non_list_saved_tensors + 2 * num_experts);

    const size_t first_intermediate_fwd_output =
        non_list_saved_tensors + weights_per_expert * num_experts;
    const auto& chunks_input = saved_vars[first_intermediate_fwd_output];
    const auto& token_to_chunk = saved_vars[first_intermediate_fwd_output + 1];
    const auto& token_in_chunk = saved_vars[first_intermediate_fwd_output + 2];
    const auto& chunks_routing_table =
        saved_vars[first_intermediate_fwd_output + 3];
    const auto& gemm12_out = saved_vars[first_intermediate_fwd_output + 4];
    const auto& activation_out = saved_vars[first_intermediate_fwd_output + 5];
    const auto& mult_out = saved_vars[first_intermediate_fwd_output + 6];

    auto result = mixture_of_experts_bwd_fused_weights(
        grads[0],
        saved_vars[0],
        chunks_input,
        token_to_chunk,
        token_in_chunk,
        chunks_routing_table,
        gemm12_out,
        activation_out,
        mult_out,
        w12,
        w3,
        ctx->saved_data["permuted_weights"].toBool(),
        ctx->saved_data["activation"].toStringRef(),
        ctx->saved_data["experts_min"].toInt(),
        ctx->saved_data["experts_max"].toInt());

    const size_t num_fwd_inputs =
        non_list_inputs + weights_per_expert * num_experts;
    torch::autograd::variable_list grad_input(num_fwd_inputs, at::Tensor());

    grad_input[0] = result[0];
    for (size_t i = 0; i < num_experts; i++) {
      grad_input[non_list_tensors + i] = result[1 + i];
      grad_input[non_list_tensors + num_experts + i] =
          result[1 + num_experts + i];
    }
    return grad_input;
  }

 private:
  static const size_t weights_per_expert = 2;
  static const size_t non_list_saved_tensors = 1;
  static const size_t non_list_tensors = 3;
  static const size_t non_list_inputs = 7;
  static const size_t outputs_for_bwd = 7;
};

class MixtureOfExpertsRecompFusedWeightsFunction
    : public torch::autograd::Function<
          MixtureOfExpertsRecompFusedWeightsFunction> {
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

    size_t num_experts = w12.size();
    torch::autograd::variable_list to_save;
    to_save.reserve(non_list_tensors + weights_per_expert * num_experts);
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

    return {output};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grads) {
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
        non_list_inputs + weights_per_expert * num_experts;
    torch::autograd::variable_list grad_input(num_fwd_inputs, at::Tensor());

    grad_input[0] = result[0];
    for (size_t i = 0; i < num_experts; i++) {
      grad_input[non_list_tensors + i] = result[1 + i];
      grad_input[non_list_tensors + num_experts + i] =
          result[1 + num_experts + i];
    }
    return grad_input;
  }

 private:
  static const size_t weights_per_expert = 2;
  static const size_t non_list_tensors = 3;
  static const size_t non_list_inputs = 7;
};

} // namespace habana
