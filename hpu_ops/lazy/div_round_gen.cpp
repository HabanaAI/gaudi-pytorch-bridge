/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "hpu_ops/common/div_round_gen.h"
#include "backend/synapse_helpers/device_helpers.h"
#include "generated/lazy/div.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/binary_kernels.h"
#include "hpu_ops/common/div_round_gen.h"
#include "hpu_ops/div_mod_util.h"

namespace habana {

static void convert_scalar_to_tensor(
    at::Stack& stack,
    c10::optional<c10::ScalarType> compute_dtype = c10::nullopt) {
  auto& other_ival = stack.at(1);
  const auto& other = other_ival.toScalar();
  other_ival = habana_lazy::get_tensor_for_scalar(
      other.to<double>(), compute_dtype.value_or(other.type()));
}

static bool DivCommonCheck(
    const at::Tensor& self,
    const c10::IValue& other,
    c10::optional<c10::string_view>&& rounding_mode) {
  auto promote_int_to_float = !rounding_mode;
  auto result_type = GetCommonDtype({self, other}, promote_int_to_float);

  switch (result_type) {
    case torch::kBFloat16:
    case torch::kFloat32:
    case torch::kFloat64:
      return true;
    case torch::kHalf: {
      return synapse_helpers::device_supports_fp16(
          HPURegistrar::get_device().type());
    }
    case torch::kInt8:
    case torch::kInt16:
    case torch::kInt32:
    case torch::kInt64:
      // floor and trunc support integral types by casts
      return rounding_mode.has_value();
    default:
      return false;
  }
}

FALLBACK_CHECK(
    DivTensorModeFallbackCheck,
    const at::Tensor& self,
    const at::Tensor& other,
    c10::optional<c10::string_view> rounding_mode) {
  return DivCommonCheck(self, other, std::move(rounding_mode));
}

FALLBACK_CHECK(
    DivScalarModeFallbackCheck,
    const at::Tensor& self,
    const at::Scalar& other,
    c10::optional<c10::string_view> rounding_mode) {
  return DivCommonCheck(self, other, std::move(rounding_mode));
}

template <>
LazyDivScalarInplace<at::Tensor&>::LazyDivScalarInplace(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  convert_scalar_to_tensor(get_inputs());
}
template <>
at::Tensor& LazyDivScalarInplace<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

template <typename T>
static void div_mode(habana_lazy::LazyOp<T>* op, at::Stack& inputs) {
  c10::optional<c10::string_view> rounding_mode =
      inputs.at(2).toOptional<c10::string_view>();
  TORCH_CHECK(
      !rounding_mode.has_value() or (*rounding_mode == "trunc") or
          (*rounding_mode == StrModeFloor),
      "div expected rounding_mode to be one of None, '",
      StrModeTruncate,
      "', or '",
      StrModeFloor,
      "' "
      "but found '",
      *rounding_mode,
      "'");
  at::ScalarType result_type =
      GetResultDtype(inputs, !rounding_mode.has_value());
  op->set_scalar_types({result_type});
  if (inputs.at(1).isScalar()) {
    convert_scalar_to_tensor(inputs, result_type);
  }
}

template <>
DivMode<at::Tensor>::DivMode(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  div_mode(this, get_inputs());
}

template <>
at::Tensor DivMode<at::Tensor>::get_result_overrideable() {
  return LazyOp<at::Tensor>::get_result_overrideable();
}

template <>
DivMode<at::Tensor&>::DivMode(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor&>(qualstring, inputs, out_shapes_fn) {
  div_mode(this, get_inputs());
}

template <>
at::Tensor& DivMode<at::Tensor&>::get_result_overrideable() {
  return LazyOp<at::Tensor&>::get_result_overrideable();
}

} // namespace habana
