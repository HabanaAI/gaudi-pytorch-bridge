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

#include "generated/autograd/autograd_ops.h"

namespace habana {

std::vector<at::Tensor> mixture_of_experts_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    at::TensorList w1,
    at::TensorList w2,
    at::TensorList w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max);

at::Tensor mixture_of_experts_recomp_fwd(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_fwd_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

at::Tensor mixture_of_experts_recomp_fwd_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& chunks_routing_weights,
    const at::Tensor& gemm1_out,
    const at::Tensor& gemm2_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::Tensor& mlp_out,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size);

std::vector<at::Tensor> mixture_of_experts_recomp_bwd(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

std::vector<at::Tensor> mixture_of_experts_bwd_fused_weights(
    const at::Tensor& grad_tokens_in,
    const at::Tensor& chunks_input,
    const at::Tensor& token_to_chunk,
    const at::Tensor& token_in_chunk,
    const at::Tensor& chunks_routing_table,
    const at::Tensor& chunks_routing_weights,
    const at::Tensor& gemm12_out,
    const at::Tensor& activation_out,
    const at::Tensor& mult_out,
    const at::Tensor& mlp_out,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size);

std::vector<at::Tensor> mixture_of_experts_recomp_bwd_fused_weights(
    const at::Tensor& grad,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

constexpr size_t weights_per_expert = 3;
constexpr size_t weights_per_expert_fused = 2;
constexpr size_t non_list_tensors = 3;
constexpr size_t non_list_inputs = 7;
constexpr size_t outputs_for_bwd = 10;
constexpr size_t outputs_for_bwd_fused = 9;

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
    const std::optional<bool> recomp);

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
    const std::optional<bool> recomp);

} // namespace habana
