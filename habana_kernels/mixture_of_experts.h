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
#include <ATen/core/Tensor.h>

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
    const int64_t experts_max);

at::Tensor mixture_of_experts_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

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
    const bool measurement_mode);

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
    const bool measurement_mode);

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
    const int64_t experts_max);

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
    const int64_t experts_max);

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
    const int64_t experts_max);

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
    const int64_t experts_max);

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
    const int64_t experts_max);

at::Tensor mixture_of_experts_fwd_fused_weights_autograd_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const bool permuted_weights,
    const c10::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max);

} // namespace habana_lazy
