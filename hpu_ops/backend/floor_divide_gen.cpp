/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/floor_divide.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {
std::shared_ptr<void> FillFloorDivideParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_DivModKernel::ParamsV2);
  // using floor mode
  params->isTruncRoundingMode = false;
  return params;
}

SharedMetaDataVector FloorDivideSharedMeta(const at::Stack& stack) {
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
