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
#include "habana_helpers/logging_pt.h"

namespace habana {

namespace {

template <typename T>
const char* getOverloadName() {
  if constexpr (std::is_same_v<T, const std::optional<at::Tensor>&>) {
    return "";
  } else if constexpr (std::is_same_v<T, double>) {
    return "scalar";
  } else if constexpr (std::is_same_v<T, at::ArrayRef<double>>) {
    return "scalar_list";
  }
  HABANA_ASSERT(
      false,
      "Unsupported type for overload name. Only const std::optional<at::Tensor>&, double, and at::ArrayRef<double> are supported.");
}

template <typename T>
at::Tensor cast_from_fp8_dispatch(
    const at::Tensor& input,
    T scale,
    at::ScalarType out_dtype,
    at::OptionalIntArrayRef scale_shape) {
  static const char* overload_name = getOverloadName<T>();
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::cast_from_fp8", overload_name)
          .typed<at::Tensor(
              const at::Tensor&, T, at::ScalarType, at::OptionalIntArrayRef)>();

  return op.call(input, scale, out_dtype, scale_shape);
}

template <typename T>
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_fp8_gemm_bwd_dispatch(
    const at::Tensor& grad_in,
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    T A_scale_inv,
    T B_scale_inv,
    bool has_bias,
    bool has_accumulate) {
  static const char* overload_name = getOverloadName<T>();
  static auto op =
      torch::Dispatcher::singleton()
          .findSchemaOrThrow("hpu::_fp8_gemm_bwd", overload_name)
          .typed<std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
              const at::Tensor&,
              const at::Tensor&,
              bool,
              const at::Tensor&,
              bool,
              T,
              T,
              bool,
              bool)>();
  return op.call(
      grad_in,
      A,
      trans_A,
      B,
      trans_B,
      A_scale_inv,
      B_scale_inv,
      has_bias,
      has_accumulate);
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
  if (scale.has_value()) {
    ctx->save_for_backward({scale.value()});
  }
  ctx->saved_data["out_dtype"] = input.scalar_type();

  auto result = cast_to_fp8_v2_dispatch(
      input, scale, stochastic_rounding, is_amax, dtype, scale_shape);
  return {std::get<0>(result), std::get<1>(result)};
}

std::vector<at::Tensor> CastToFp8V2Function::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grads) {
  auto saved_data = ctx->get_saved_variables();
  std::optional<at::Tensor> scale = std::nullopt;
  if (saved_data.size() >= 1) {
    scale = saved_data.at(0);
  }
  ScalarType out_dtype = ctx->saved_data["out_dtype"].toScalarType();

  auto result = cast_from_fp8_dispatch<const std::optional<at::Tensor>&>(
      grads[0], scale, out_dtype, std::nullopt);

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
      cast_from_fp8_dispatch(grads[0], scale, out_dtype, std::nullopt);

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
  auto scale = at::ArrayRef<double>(scale_vec);
  at::ScalarType out_dtype = ctx->saved_data["out_dtype"].toScalarType();

  auto result =
      cast_from_fp8_dispatch(grads[0], scale, out_dtype, scale.size());

  return {
      result,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

std::vector<at::Tensor> processBwdResults(
    const std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> results,
    const bool has_bias,
    const bool has_D) {
  auto grad_A = std::get<0>(results);
  auto grad_B = std::get<1>(results);

  auto grad_bias = at::Tensor();
  if (has_bias) {
    grad_bias = std::get<2>(results);
  }

  auto grad_D = at::Tensor();
  if (has_D) {
    grad_D = std::get<3>(results);
  }

  return {
      grad_A,
      at::Tensor(),
      grad_B,
      at::Tensor(),
      grad_D,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      grad_bias,
      at::Tensor(),
      at::Tensor()};
}

at::Tensor Fp8GemmV2Function::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const std::optional<at::Tensor>& D,
    ScalarType out_dtype,
    const std::optional<at::Tensor>& A_scale_inv,
    const std::optional<at::Tensor>& B_scale_inv,
    const std::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  at::AutoDispatchBelowADInplaceOrView g;
  ctx->saved_data["trans_A"] = trans_A;
  ctx->saved_data["trans_B"] = trans_B;
  ctx->saved_data["has_D"] =
      D.has_value() && accumulate && D.value().requires_grad();
  ctx->saved_data["has_A_scale_inv"] = A_scale_inv.has_value();
  ctx->saved_data["has_B_scale_inv"] = B_scale_inv.has_value();
  ctx->saved_data["has_bias"] =
      bias.has_value() && bias.value().requires_grad();

  std::vector<at::Tensor> values_for_backward = {A, B};
  if (A_scale_inv.has_value()) {
    values_for_backward.push_back(A_scale_inv.value());
  }
  if (B_scale_inv.has_value()) {
    values_for_backward.push_back(B_scale_inv.value());
  }
  ctx->save_for_backward(values_for_backward);

  return fp8_gemm_v2_dispatch(
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
      B_scale_shape);
}

std::vector<at::Tensor> Fp8GemmV2Function::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad) {
  bool trans_A = ctx->saved_data["trans_A"].toBool();
  bool trans_B = ctx->saved_data["trans_B"].toBool();
  bool has_D = ctx->saved_data["has_D"].toBool();

  bool has_A_scale_inv = ctx->saved_data["has_A_scale_inv"].toBool();
  bool has_B_scale_inv = ctx->saved_data["has_B_scale_inv"].toBool();
  bool has_bias = ctx->saved_data["has_bias"].toBool();

  auto fwdTensors = ctx->get_saved_variables();
  int fwdTensorsIdx = 0;

  auto grad_in = grad[0];
  auto A = fwdTensors[fwdTensorsIdx++];
  auto B = fwdTensors[fwdTensorsIdx++];

  std::optional<at::Tensor> A_scale_inv{};
  if (has_A_scale_inv) {
    A_scale_inv = fwdTensors[fwdTensorsIdx++];
  }

  std::optional<at::Tensor> B_scale_inv{};
  if (has_B_scale_inv) {
    B_scale_inv = fwdTensors[fwdTensorsIdx++];
  }

  auto results = _fp8_gemm_bwd_dispatch<const std::optional<at::Tensor>&>(
      grad_in,
      A,
      trans_A,
      B,
      trans_B,
      A_scale_inv,
      B_scale_inv,
      has_bias,
      has_D);

  return processBwdResults(results, has_bias, has_D);
}

at::Tensor Fp8GemmV2ScalarFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const std::optional<at::Tensor>& D,
    ScalarType out_dtype,
    double A_scale_inv,
    double B_scale_inv,
    const std::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  at::AutoDispatchBelowADInplaceOrView g;

  ctx->saved_data["trans_A"] = trans_A;
  ctx->saved_data["trans_B"] = trans_B;
  ctx->saved_data["has_D"] =
      D.has_value() && accumulate && D.value().requires_grad();
  ctx->saved_data["A_scale_inv"] = A_scale_inv;
  ctx->saved_data["B_scale_inv"] = B_scale_inv;
  ctx->saved_data["has_bias"] =
      bias.has_value() && bias.value().requires_grad();

  ctx->save_for_backward({A, B});

  return fp8_gemm_v2_scalar_dispatch(
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
      B_scale_shape);
}

std::vector<at::Tensor> Fp8GemmV2ScalarFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad) {
  bool trans_A = ctx->saved_data["trans_A"].toBool();
  bool trans_B = ctx->saved_data["trans_B"].toBool();
  bool has_D = ctx->saved_data["has_D"].toBool();
  double A_scale_inv = ctx->saved_data["A_scale_inv"].toDouble();
  double B_scale_inv = ctx->saved_data["B_scale_inv"].toDouble();
  bool has_bias = ctx->saved_data["has_bias"].toBool();

  auto fwdTensors = ctx->get_saved_variables();

  auto grad_in = grad[0];
  auto A = fwdTensors[0];
  auto B = fwdTensors[1];

  auto results = _fp8_gemm_bwd_dispatch(
      grad_in,
      A,
      trans_A,
      B,
      trans_B,
      A_scale_inv,
      B_scale_inv,
      has_bias,
      has_D);

  return processBwdResults(results, has_bias, has_D);
}

at::Tensor Fp8GemmV2ScalarListFunction::forward(
    torch::autograd::AutogradContext* ctx,
    const at::Tensor& A,
    bool trans_A,
    const at::Tensor& B,
    bool trans_B,
    const std::optional<at::Tensor>& D,
    ScalarType out_dtype,
    ArrayRef<double> A_scale_inv,
    ArrayRef<double> B_scale_inv,
    const std::optional<at::Tensor>& bias,
    bool accumulate,
    OptionalIntArrayRef B_scale_shape) {
  at::AutoDispatchBelowADInplaceOrView g;

  ctx->saved_data["trans_A"] = trans_A;
  ctx->saved_data["trans_B"] = trans_B;
  ctx->saved_data["has_D"] =
      D.has_value() && accumulate && D.value().requires_grad();
  ctx->saved_data["A_scale_inv"] = A_scale_inv.vec();
  ctx->saved_data["B_scale_inv"] = B_scale_inv.vec();
  ctx->saved_data["has_bias"] =
      bias.has_value() && bias.value().requires_grad();

  ctx->save_for_backward({A, B});

  return fp8_gemm_v2_scalar_list_dispatch(
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
      B_scale_shape);
}

std::vector<at::Tensor> Fp8GemmV2ScalarListFunction::backward(
    torch::autograd::AutogradContext* ctx,
    const std::vector<at::Tensor>& grad) {
  bool trans_A = ctx->saved_data["trans_A"].toBool();
  bool trans_B = ctx->saved_data["trans_B"].toBool();
  bool has_D = ctx->saved_data["has_D"].toBool();
  std::vector<double> A_scale_inv_vec =
      ctx->saved_data["A_scale_inv"].toDoubleVector();
  auto A_scale_inv = at::ArrayRef<double>(A_scale_inv_vec);

  std::vector<double> B_scale_inv_vec =
      ctx->saved_data["B_scale_inv"].toDoubleVector();
  auto B_scale_inv = at::ArrayRef<double>(B_scale_inv_vec);

  bool has_bias = ctx->saved_data["has_bias"].toBool();

  auto fwdTensors = ctx->get_saved_variables();
  auto grad_in = grad[0];
  auto A = fwdTensors[0];
  auto B = fwdTensors[1];

  auto results = _fp8_gemm_bwd_dispatch(
      grad_in,
      A,
      trans_A,
      B,
      trans_B,
      A_scale_inv,
      B_scale_inv,
      has_bias,
      has_D);

  return processBwdResults(results, has_bias, has_D);
}

} // namespace habana
