/*******************************************************************************
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
#include "backend/helpers/lowering_util.h"
#include <ATen/native/ReduceOpsUtils.h>

namespace habana {

void LoweringUtil::SortAndRemoveDuplicateDims(
    std::vector<int64_t>& in_dim,
    int64_t ndim) {
  // TODO:Any maybe_wrap_dim(...) to be removed after
  // https://gerrit.habana-labs.com/#/c/141381/ is merged.
  for (auto iterator_dim = in_dim.begin(); iterator_dim != in_dim.end();
       ++iterator_dim) {
    *iterator_dim = c10::maybe_wrap_dim(*iterator_dim, ndim);
  }

  std::sort(in_dim.begin(), in_dim.end());
  auto last = std::unique(in_dim.begin(), in_dim.end());
  in_dim.erase(last, in_dim.end());
}

std::vector<int64_t> LoweringUtil::ComputeOutputShape(
    const at::Tensor& self,
    const at::IntArrayRef dim,
    const bool keepdim) {
  DimMask dim_mask = MakeDimMask(dim, self.dim());
  std::vector<int64_t> shape = self.sizes().vec();
  for (int64_t dimIndex = shape.size() - 1; dimIndex >= 0; dimIndex--) {
    if (dim_mask[dimIndex]) {
      if (keepdim) {
        shape[dimIndex] = 1;
      } else {
        shape.erase(shape.begin() + dimIndex);
      } // if (keepdim)
    }
  } // for (int64_t
  return shape;
}

c10::ScalarType LoweringUtil::GetDtype(
    at::Tensor& result,
    const at::Tensor& self,
    c10::optional<c10::ScalarType> dtype,
    bool promote_integers) {
  if (dtype.has_value()) {
    return dtype.value();

  } else if (result.defined()) {
    return result.scalar_type();
  }
  c10::ScalarType src_type = self.scalar_type();
  if (promote_integers && at::isIntegralType(src_type, /*includeBool=*/true)) {
    return torch::kLong;
  }
  return src_type;
}

DimMask LoweringUtil::MakeDimMask(at::IntArrayRef dims, int64_t ndim) {
  return at::native::make_dim_mask(dims, ndim);
}

} // namespace habana
