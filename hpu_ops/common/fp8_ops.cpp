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

#include <torch/torch.h>
#include "generated/autograd/autograd_ops.h"

namespace habana {

namespace {

at::Tensor cast_from_fp8_dispatch(
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale,
    at::ScalarType out_dtype,
    at::OptionalIntArrayRef scale_shape) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::cast_from_fp8", "")
                       .typed<decltype(cast_from_fp8_dispatch)>();

  return op.call(input, scale, out_dtype, scale_shape);
}

at::Tensor cast_from_fp8_scalar_dispatch(
    const at::Tensor& input,
    double scale,
    at::ScalarType out_dtype,
    at::OptionalIntArrayRef scale_shape) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::cast_from_fp8", "scalar")
                       .typed<decltype(cast_from_fp8_scalar_dispatch)>();

  return op.call(input, scale, out_dtype, scale_shape);
}

at::Tensor cast_from_fp8_scalar_list_dispatch(
    const at::Tensor& input,
    at::ArrayRef<double> scale,
    at::ScalarType out_dtype,
    at::OptionalIntArrayRef scale_shape) {
  static auto op = torch::Dispatcher::singleton()
                       .findSchemaOrThrow("hpu::cast_from_fp8", "scalar_list")
                       .typed<decltype(cast_from_fp8_scalar_list_dispatch)>();

  return op.call(input, scale, out_dtype, scale_shape);
}

} // namespace

std::vector<at::Tensor> CastToFp8V2Function::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    const std::optional<at::Tensor>& scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    OptionalIntArrayRef scale_shape) {
  at::AutoDispatchBelowADInplaceOrView g;

  ctx->save_for_backward({scale.value_or(at::Tensor())});
  ctx->saved_data["out_dtype"] = input.scalar_type();

  auto result = cast_to_fp8_v2_dispatch(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
  return {std::get<0>(result), std::get<1>(result)};
}

std::vector<at::Tensor> CastToFp8V2Function::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  auto scale = ctx->get_saved_variables().at(0);
  ScalarType out_dtype = ctx->saved_data["out_dtype"].toScalarType();

  auto result =
      cast_from_fp8_dispatch(grads[0], scale, out_dtype, c10::nullopt);

  return {
      result,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

std::vector<at::Tensor> CastToFp8V2ScalarFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    double scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape) {
  at::AutoDispatchBelowADInplaceOrView g;

  ctx->saved_data["scale"] = scale;
  ctx->saved_data["out_dtype"] = input.scalar_type();

  auto result = cast_to_fp8_v2_scalar_dispatch(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
  return {std::get<0>(result), std::get<1>(result)};
}

std::vector<at::Tensor> CastToFp8V2ScalarFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  double scale = ctx->saved_data["scale"].toDouble();
  ScalarType out_dtype = ctx->saved_data["out_dtype"].toScalarType();

  auto result =
      cast_from_fp8_scalar_dispatch(grads[0], scale, out_dtype, c10::nullopt);

  return {
      result,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
};

std::vector<at::Tensor> CastToFp8V2ScalarListFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& input,
    ArrayRef<double> scale,
    bool stochastic_rounding,
    bool is_amax,
    at::ScalarType dtype,
    at::OptionalIntArrayRef scale_shape) {
  at::AutoDispatchBelowADInplaceOrView g;

  ctx->saved_data["scale"] = scale.vec();
  ctx->saved_data["out_dtype"] = input.scalar_type();

  auto result = cast_to_fp8_v2_scalar_list_dispatch(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
  return {std::get<0>(result), std::get<1>(result)};
}

std::vector<at::Tensor> CastToFp8V2ScalarListFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  std::vector<double> scale_vec = ctx->saved_data["scale"].toDoubleVector();
  at::ArrayRef<double> scale = at::ArrayRef<double>(scale_vec);
  at::ScalarType out_dtype = ctx->saved_data["out_dtype"].toScalarType();

  auto result = cast_from_fp8_scalar_list_dispatch(
      grads[0], scale, out_dtype, scale.size());

  return {
      result,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

} // namespace habana
