/**
* Copyright (c) 2021-2024 Intel Corporation
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

at::optional<at::ScalarType> get_dtype(
    const at::Stack& stack,
    at::optional<uint8_t> dtype_index) {
  return dtype_index.has_value()
      ? stack.at(dtype_index.value()).toOptional<at::ScalarType>()
      : at::nullopt;
}

std::vector<int64_t> get_dims(
    const at::Stack& stack,
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

unsigned ReductionMask(const at::Tensor& self, at::optional<int64_t> dimOpt) {
  if (!dimOpt || self.dim() == 0) {
    return 0;
  }
  int64_t dim = *dimOpt;
  int64_t dimBitPos = (dim >= 0 ? self.dim() : 0) - dim - 1;
  return 1 << dimBitPos;
}

} // namespace habana
