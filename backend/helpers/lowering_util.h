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

#pragma once
#include "backend/habana_operator.h"

namespace habana {

using DimMask = std::bitset<64>;
class LoweringUtil {
 public:
  static void SortAndRemoveDuplicateDims(
      std::vector<int64_t>& in_dim,
      int64_t ndim);

  // This can be used for Reduction type ops and Norm type Ops like
  //... norm(..., IntArrayRef dim, bool keepdim);
  static std::vector<int64_t> ComputeOutputShape(
      const at::Tensor& self,
      const at::IntArrayRef dim,
      const bool keepdim);

  static c10::ScalarType GetDtype(
      at::Tensor& result,
      const at::Tensor& self,
      c10::optional<c10::ScalarType> dtype,
      bool promote_integers = false);

  static DimMask MakeDimMask(at::IntArrayRef dims, int64_t ndim);

  static constexpr float FP_INFINITY = std::numeric_limits<float>::infinity();
  static constexpr float FP_NEG_INFINITY =
      -std::numeric_limits<float>::infinity();
};

} // namespace habana
