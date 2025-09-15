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
#include "generated/backend/narrow_copy.h"
#include "habana_helpers/conversion.h"

namespace habana {

OutputMetaDataVector NarrowCopyMeta(const at::Stack& stack) {
  const at::Tensor& self = stack[0].toTensor();
  int64_t dim_unwrapped = stack[1].toInt();
  int64_t start = stack[2].toInt();
  int64_t length = stack[3].toInt();

  auto shape = self.sizes().vec();
  const auto dim =
      static_cast<size_t>(at::maybe_wrap_dim(dim_unwrapped, self.dim()));
  TORCH_CHECK(
      dim < shape.size(),
      "NarrowCopyMeta: Dimension out of range in NarrowCopyMeta");

  int64_t dim_size = shape[dim];

  TORCH_CHECK(
      start >= -dim_size && start <= dim_size,
      "NarrowCopyMeta: start out of range. Expected to be in range of [",
      -dim_size,
      ", ",
      dim_size,
      "], but got ",
      start);

  if (start < 0) {
    start += dim_size;
  }

  TORCH_CHECK(
      length >= 0 && start + length <= dim_size,
      "NarrowCopyMeta: start (",
      start,
      ") + length (",
      length,
      ") must be <= size (",
      dim_size,
      ") at dimension ",
      dim);

  shape[dim] = length;

  OutputMetaData meta;
  meta.shape = shape;
  meta.dtype = self.scalar_type();
  return {meta};
}

FillParamsT FillNarrowCopyParams(const at::Stack& stack) {
  const at::Tensor& self = stack[0].toTensor();
  int64_t dim_unwrapped = stack[1].toInt();
  int64_t start = stack[2].toInt();
  int64_t length = stack[3].toInt();

  const auto dim =
      static_cast<size_t>(at::maybe_wrap_dim(dim_unwrapped, self.dim()));
  int64_t dim_size = self.sizes()[dim];
  if (start < 0) {
    start += dim_size;
  }

  PARAMS_STUB(ns_NarrowCopy::Params);
  params->dim = safe_convert<int>(self.sizes().size() - 1 - dim);
  params->start = safe_convert<int>(start);
  params->length = safe_convert<int>(length);
  return paramsT;
}

} // namespace habana
