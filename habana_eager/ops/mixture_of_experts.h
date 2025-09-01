/**
 * Copyright (c) 2024-2025 Intel Corporation
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

namespace habana::eager {

at::Tensor mixture_of_experts(
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
    const std::optional<bool> recomp,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::optional<bool> recomp,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_bias_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList w12_bias,
    const at::TensorList w3_bias,
    const bool permuted_weights,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0,
    const double alpha = 1.704,
    const double limit = 7.0);

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement(
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
    const bool measurement_mode,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

std::tuple<at::Tensor, at::Tensor>
mixture_of_experts_fp8_measurement_fused_weights(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool measurement_mode,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8(
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights(
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_scalars(
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights_scalars(
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
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_scalars_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_w1,
    const c10::ArrayRef<double>& d_scale_w2,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights_scalars_dynamic(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const double d_scale_hidden_states,
    const c10::ArrayRef<double>& d_scale_w12,
    const c10::ArrayRef<double>& d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_blockwise(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const int64_t block_size,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights_blockwise(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const int64_t block_size,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

} // namespace habana::eager
