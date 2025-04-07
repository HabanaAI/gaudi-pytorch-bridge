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
#include "habana_kernels/h2d_scales.h"
#include "habana_kernels/lazy_custom_op_declarations.h"
#include "habana_kernels/lazy_kernels.h"

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

  const auto h2d_scales_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES);
  LazyOp<::std::tuple<at::Tensor, at::Tensor>> hpu_op{
      "hpu::cast_to_fp8_v2",
      {input,
       maybe_convert_to_h2d(scale, h2d_scales_enabled, "cast_to_fp8_v2"sv),
       stochastic_rounding,
       is_amax,
       dtype,
       scale_shape}};
  hpu_op.SetOutputMetaFn(CastToFp8V2Meta);
  RUN_TUPLE_MAYBE_WITH_ACC_THREAD(cast_to_fp8_v2, hpu_op);
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

  const auto h2d_scales_enabled = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_SCALES);
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

} // namespace habana_lazy
