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
#include "generated/lazy/cast_to_fp8_v2.h"
#include "generated/lazy/fp8_gemm_v2.h"
#include "habana_kernels/lazy_custom_op_declarations.h"
#include "habana_kernels/lazy_kernels.h"

using namespace habana;

namespace {

at::Tensor create_h2d_scale(void* scale) {
  auto scale_tensor = habana_lazy::empty_hpu_lazy(
      {1}, at::ScalarType::Float, c10::nullopt, false, HOST_TO_DEVICE_TENSOR);

  auto hl_params_shape =
      habana_lazy::GetOrCreateHbLazyTensor(scale_tensor, c10::kHPU);
  auto hl_param_internal = hl_params_shape.CurrentTensorAttached().value();
  auto tmeta{get_tensor_extra_meta(hl_param_internal)};

  tmeta->set_host_data(scale, {1}, sizeof(float_t), HostDataType::FLOAT_T);

  return scale_tensor;
}

bool is_cpu_float_0d_tensor(const c10::optional<at::Tensor>& tensor) {
  return tensor.has_value() and tensor.value().defined() and
      tensor->device().is_cpu() and
      tensor->scalar_type() == at::ScalarType::Float and tensor->dim() == 0;
}

} // namespace

namespace habana_lazy {

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_lazy(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape) {
  PT_LAZY_TRACE;

  std::vector<at::IValue> inputs{
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape};
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES) and
      is_cpu_float_0d_tensor(scale)) {
    inputs[1] = create_h2d_scale(scale->data_ptr());
    PT_BRIDGE_DEBUG(
        "CPU Tensor scale of node hpu::cast_to_fp8_v2 was converted to H2D tensor.");
  }
  LazyOp<::std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2", std::move(inputs)};
  hpu_op.SetOutputMetaFn(CastToFp8V2Meta);
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(cast_to_fp8_v2, hpu_op);
}

std::tuple<at::Tensor, at::Tensor> cast_to_fp8_v2_scalar_lazy(
    const at::Tensor& input,
    double scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape) {
  PT_LAZY_TRACE;

  std::vector<at::IValue> inputs{
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape};
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES)) {
    float scale_float = static_cast<float>(scale);
    inputs[1] = create_h2d_scale(&scale_float);
    PT_BRIDGE_DEBUG(
        "Scalar scale of node hpu::cast_to_fp8_v2.scalar was converted to H2D tensor.");
  }
  LazyOp<std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2", std::move(inputs)};
  hpu_op.SetOutputMetaFn(CastToFp8V2Meta);
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(cast_to_fp8_v2, hpu_op);
}

at::Tensor fp8_gemm_v2_lazy(
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
    at::OptionalIntArrayRef B_scale_shape) {
  PT_LAZY_TRACE;

  std::vector<at::IValue> inputs{
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
      B_scale_shape};
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES) and
      is_cpu_float_0d_tensor(A_scale_inv) and
      is_cpu_float_0d_tensor(B_scale_inv)) {
    inputs[6] = create_h2d_scale(A_scale_inv->data_ptr());
    inputs[7] = create_h2d_scale(B_scale_inv->data_ptr());
    PT_BRIDGE_DEBUG(
        "CPU Tensor scales of node hpu::fp8_gemm_v2 were converted to H2D tensors.");
  }
  LazyOp<at::Tensor> hpu_op{"hpu::fp8_gemm_v2", std::move(inputs)};
  hpu_op.SetOutputMetaFn(Fp8GemmV2Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fp8_gemm_v2, hpu_op);
}

at::Tensor fp8_gemm_v2_scalar_lazy(
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
    at::OptionalIntArrayRef B_scale_shape) {
  PT_LAZY_TRACE;

  std::vector<at::IValue> inputs{
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
      B_scale_shape};
  if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES)) {
    float scale_a_float = static_cast<float>(A_scale_inv);
    float scale_b_float = static_cast<float>(B_scale_inv);
    inputs[6] = create_h2d_scale(&scale_a_float);
    inputs[7] = create_h2d_scale(&scale_b_float);
    PT_BRIDGE_DEBUG(
        "Scalar scales of node hpu::fp8_gemm_v2.scalar were converted to H2D tensors.");
  }
  LazyOp<at::Tensor> hpu_op{"hpu::fp8_gemm_v2", std::move(inputs)};
  hpu_op.SetOutputMetaFn(Fp8GemmV2Meta);
  RUN_MAYBE_WITH_ACC_THREAD(fp8_gemm_v2, hpu_op);
}

} // namespace habana_lazy
