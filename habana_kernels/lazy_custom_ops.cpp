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
#include "generated/lazy/cast_from_fp8.h"
#include "generated/lazy/cast_to_fp8_v2.h"
#include "generated/lazy/conv2d_fp8.h"
#include "generated/lazy/fp8_gemm_v2.h"
#include "generated/lazy/mixture_of_experts.h"
#include "habana_kernels/h2d_scales_lazy.h"
#include "habana_kernels/lazy_custom_op_declarations.h"
#include "habana_kernels/lazy_kernels.h"
#include "pytorch_helpers/habana_helpers/h2d_scales.h"

using namespace habana;

namespace habana_lazy {

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_lazy(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape) {
  PT_LAZY_TRACE;

  LazyOp<::std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2",
      {input,
       maybe_convert_to_h2d(
           scale, habana_helpers::is_h2d_scales_enabled(), "cast_to_fp8_v2"sv),
       stochastic_rounding,
       is_amax,
       dtype,
       scale_shape}};
  hpu_op.SetOutputMetaFn(CastToFp8V2Meta);
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(cast_to_fp8_v2, hpu_op);
}

at::Tensor cast_from_fp8_lazy(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape) {
  PT_LAZY_TRACE;

  LazyOp<at::Tensor> hpu_op{
      "hpu::cast_from_fp8",
      {input,
       maybe_convert_to_h2d(
           scale, habana_helpers::is_h2d_scales_enabled(), "cast_from_fp8"sv),
       dtype,
       scale_shape}};
  hpu_op.SetOutputMetaFn(CastFromFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(cast_from_fp8, hpu_op);
}

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
    const std::optional<at::Tensor>& scale_weight) {
  PT_LAZY_TRACE;

  const bool is_cpu_scale =
      (scale_input.has_value() and scale_input->is_cpu()) or
      (scale_weight.has_value() and scale_weight->is_cpu());
  HABANA_ASSERT(
      not is_cpu_scale,
      "conv2d_fp8.default doesn't support H2D scales feature yet, but received CPU scales.");

  LazyOp<at::Tensor> hpu_op{
      "hpu::conv2d_fp8",
      {input,
       weight,
       bias,
       stride,
       padding,
       dilation,
       groups,
       out_dtype,
       scale_input,
       scale_weight}};
  hpu_op.SetOutputMetaFn(Conv2dFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(conv2d_fp8, hpu_op);
}

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
    at::OptionalIntArrayRef B_scale_shape) {
  PT_LAZY_TRACE;

  const auto h2d_scales_enabled = habana_helpers::is_h2d_scales_enabled();
  const std::string_view op_name{"fp8_gemm_v2"};
  LazyOp<at::Tensor> hpu_op{
      "hpu::fp8_gemm_v2",
      {A,
       trans_A,
       B,
       trans_B,
       D,
       out_dtype,
       maybe_convert_to_h2d(A_scale_inv, h2d_scales_enabled, op_name),
       maybe_convert_to_h2d(B_scale_inv, h2d_scales_enabled, op_name),
       bias,
       accumulate,
       B_scale_shape}};
  hpu_op.SetOutputMetaFn(Fp8GemmV2Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fp8_gemm_v2, hpu_op);
}

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
    int64_t experts_max) {
  PT_LAZY_TRACE;

  using namespace std::literals;
  verify_no_h2d_scales(
      {d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w1,
       d_scale_w2,
       d_scale_w3},
      "mixture_of_experts.fp8"sv);

  LazyOp<at::Tensor> hpu_op{
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
       experts_max}};
  hpu_op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, hpu_op);
}

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
    int64_t experts_max) {
  PT_LAZY_TRACE;

  using namespace std::literals;
  verify_no_h2d_scales(
      {d_scale_hidden_states,
       d_scale_intermediate_hidden_states,
       d_scale_w12,
       d_scale_w3},
      "mixture_of_experts.fp8_fused_weights"sv);

  LazyOp<at::Tensor> hpu_op{
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
       experts_max}};
  hpu_op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, hpu_op);
}

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
    int64_t experts_max) {
  PT_LAZY_TRACE;

  using namespace std::literals;
  verify_no_h2d_scales(
      {d_scale_hidden_states, d_scale_w1, d_scale_w2, d_scale_w3},
      "mixture_of_experts.fp8_dynamic"sv);

  LazyOp<at::Tensor> hpu_op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w1,
       w2,
       w3,
       d_scale_hidden_states,
       d_scale_w1,
       d_scale_w2,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max}};
  hpu_op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, hpu_op);
}

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
    int64_t experts_max) {
  PT_LAZY_TRACE;

  using namespace std::literals;
  verify_no_h2d_scales(
      {d_scale_hidden_states, d_scale_w12, d_scale_w3},
      "mixture_of_experts.fp8_fused_weights_dynamic"sv);

  LazyOp<at::Tensor> hpu_op{
      "hpu::mixture_of_experts",
      {hidden_states,
       expert_routing_table,
       router_weights,
       w12,
       w3,
       d_scale_hidden_states,
       d_scale_w12,
       d_scale_w3,
       permuted_weights,
       activation,
       experts_min,
       experts_max}};
  hpu_op.SetOutputMetaFn(MixtureOfExpertsFp8Meta);
  RUN_MAYBE_WITH_ACC_THREAD(mixture_of_experts, hpu_op);
}

} // namespace habana_lazy
