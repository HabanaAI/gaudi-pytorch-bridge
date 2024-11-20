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
