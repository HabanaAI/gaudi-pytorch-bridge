/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
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
