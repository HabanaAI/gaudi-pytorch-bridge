/******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/common/reduction_template.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

static at::IntArrayRef optional_to_arrayref(const c10::optional<int64_t>& opt) {
  return opt.has_value() ? opt.value() : at::IntArrayRef{};
}

static at::IntArrayRef optional_to_arrayref(
    const c10::OptionalIntArrayRef& opt) {
  return opt.has_value() ? opt.value() : at::IntArrayRef{};
}

std::vector<int64_t> get_dims(
    at::Stack stack,
    at::optional<uint8_t> dim_index) {
  std::vector<int64_t> dims;
  auto dim_ival =
      dim_index.has_value() ? stack.at(dim_index.value()) : at::IValue();
  if (dim_ival.isInt()) {
    dims = {dim_ival.toInt()};
  } else if (dim_ival.isIntList()) {
    dims = dim_ival.toIntVector();
  } else {
    HABANA_ASSERT(
        dim_ival.isNone(),
        "Reduction op dims can be int, int list or none but got ",
        dim_ival.tagKind());
  }
  return dims;
}

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::OptionalIntArrayRef dims,
    bool keepdim) {
  at::DimVector shape =
      at::meta::get_reduction_shape(self, optional_to_arrayref(dims), keepdim);
  return {std::vector<int64_t>(shape.begin(), shape.end())};
}

sizes_vec ReductionOutputShape(
    const at::Tensor& self,
    at::optional<int64_t> dims,
    bool keepdim) {
  return ReductionOutputShape(self, optional_to_arrayref(dims), keepdim);
}

template <>
at::Tensor CommonReductionFrontendTemplate<at::Tensor>::CreateResult(
    const at::Stack& stack,
    at::ScalarType dtype) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  return at::native::create_reduction_result(
      self,
      get_dims(stack, m_dim_index),
      get_keepdim(stack, m_keepdim_index),
      dtype);
}
} // namespace habana
