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
#include <ATen/Tensor.h>

namespace habana_lazy {

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_lazy(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape);

at::Tensor cast_from_fp8_lazy(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape);

at::Tensor conv2d_fp8_lazy(
    const at::Tensor& input,
    const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride,
    c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation,
    int64_t groups,
    std::optional<at::ScalarType> out_dtype,
    const std::optional<at::Tensor>& scale_input,
    const std::optional<at::Tensor>& scale_weight);

at::Tensor fp8_gemm_v2_lazy(
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const std::optional<at::Tensor>& D,
    at::ScalarType out_dtype,
    const std::optional<at::Tensor>& A_scale_inv,
    const std::optional<at::Tensor>& B_scale_inv,
    const std::optional<at::Tensor>& bias,
    bool accumulate,
    at::OptionalIntArrayRef B_scale_shape);

at::Tensor mixture_of_experts_fp8_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    at::TensorList w1,
    at::TensorList w2,
    at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    at::TensorList d_scale_intermediate_hidden_states,
    at::TensorList d_scale_w1,
    at::TensorList d_scale_w2,
    at::TensorList d_scale_w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    at::TensorList w12,
    at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    at::TensorList d_scale_intermediate_hidden_states,
    at::TensorList d_scale_w12,
    at::TensorList d_scale_w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_dynamic_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    at::TensorList w1,
    at::TensorList w2,
    at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    at::TensorList d_scale_w1,
    at::TensorList d_scale_w2,
    at::TensorList d_scale_w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

at::Tensor mixture_of_experts_fp8_fused_weights_dynamic_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    at::TensorList w12,
    at::TensorList w3,
    const at::Tensor& d_scale_hidden_states,
    at::TensorList d_scale_w12,
    at::TensorList d_scale_w3,
    bool permuted_weights,
    std::string_view activation,
    int64_t experts_min,
    int64_t experts_max,
    const int64_t chunk_size = 0,
    const int64_t total_experts = 0);

} // namespace habana_lazy
