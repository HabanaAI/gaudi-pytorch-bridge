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

#include "generated/backend/floor_divide.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {
std::shared_ptr<void> FillFloorDivideParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_DivModKernel::ParamsV2);
  // using floor mode
  params->isTruncRoundingMode = false;
  return params;
}

SharedMetaDataVector FloorDivideSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack.at(0).toTensor();
  const auto& other = stack.at(1).toTensor();
  const auto dtype = habana_helpers::DTypeHelper::binary_op_with_type_promotion(
                         {self, other}, c10::nullopt, false)
                         .get_result_dtype();
  const auto selfDim = self.dim();
  const auto otherDim = other.dim();

  SharedMetaData floorDivideMeta("floor_divide_fwd");
  floorDivideMeta.inputs_data.emplace_back(selfDim, dtype);
  floorDivideMeta.inputs_data.emplace_back(otherDim, dtype);
  floorDivideMeta.outputs_data.emplace_back(std::max(selfDim, otherDim), dtype);
  return {floorDivideMeta};
}
} // namespace habana
