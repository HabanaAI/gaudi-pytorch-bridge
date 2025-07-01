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

std::tuple<at::Tensor, at::Tensor> mixture_of_experts_fp8_measurement_lazy(
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
mixture_of_experts_fp8_measurement_fused_weights_lazy(
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

std::vector<at::Tensor> mixture_of_experts_fwd_fp8_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_recomp_fwd_fp8_fused_weights_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_bwd_fp8_fused_weights_lazy(
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
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const at::TensorList d_scale_first_gemm_grad,
    const at::TensorList d_scale_second_gemm_grad,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_recomp_bwd_fp8_fused_weights_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w12,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w12,
    const at::TensorList d_scale_w3,
    const at::TensorList d_scale_first_gemm_grad,
    const at::TensorList d_scale_second_gemm_grad,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_fwd_fp8_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_recomp_fwd_fp8_lazy(
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_bwd_fp8_lazy(
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
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const at::TensorList d_scale_mult_grad,
    const at::TensorList d_scale_activation_grad,
    const at::TensorList d_scale_second_gemm_grad,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const std::vector<int64_t> router_weights_size,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

std::vector<at::Tensor> mixture_of_experts_recomp_bwd_fp8_lazy(
    const at::Tensor& grad_out,
    const at::Tensor& hidden_states,
    const at::Tensor& expert_routing_table,
    const at::Tensor& router_weights,
    const at::TensorList w1,
    const at::TensorList w2,
    const at::TensorList w3,
    const at::TensorList d_scale_hidden_states,
    const at::TensorList d_scale_intermediate_hidden_states,
    const at::TensorList d_scale_w1,
    const at::TensorList d_scale_w2,
    const at::TensorList d_scale_w3,
    const at::TensorList d_scale_mult_grad,
    const at::TensorList d_scale_activation_grad,
    const at::TensorList d_scale_second_gemm_grad,
    const bool permuted_weights,
    const std::string_view activation,
    const int64_t experts_min,
    const int64_t experts_max,
    const bool scaled_swiglu,
    const bool hybrid_mode,
    const bool is_first_amax,
    const bool is_second_amax);

} // namespace habana_lazy
