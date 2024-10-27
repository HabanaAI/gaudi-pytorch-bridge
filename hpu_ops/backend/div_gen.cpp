/******************************************************************************
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

#include "backend/synapse_helpers/env_flags.h"
#include "generated/backend/div.h"
#include "hpu_ops/common/div_round_gen.h"

namespace habana {

SharedMetaDataVector DivideSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack.at(0);
  auto selfTensor = self.toTensor();
  auto selfRank = selfTensor.dim();
  auto selfType = selfTensor.scalar_type();
  auto other = stack.at(1);
  int64_t otherRank;
  if (other.isTensor()) {
    otherRank = other.toTensor().dim();
  } else {
    otherRank = 1;
  }

  if (c10::isIntegralType(selfType, true))
    selfType = GetCommonDtype({self, other}, true);

  auto resultType = GetResultDtype({self, other}, true);
  std::string guid = "div";
  if (selfType == at::ScalarType::Float &&
      IS_ENV_FLAG_DEFINED_NEW(PT_HPU_ENABLE_DIV_PRECISE) &&
      GET_ENV_FLAG_NEW(PT_HPU_ENABLE_DIV_PRECISE))
    guid = "div_precise";

  SharedMetaData divMeta{guid};
  divMeta.inputs_data = {{selfRank, resultType}, {otherRank, resultType}};
  divMeta.outputs_data = {{std::max(selfRank, otherRank), resultType}};
  return {divMeta};
}

void Divide::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  if (ScalarType() == at::ScalarType::Float) {
    // Update the div guid with precise based on env
    update_div_guid_with_precise(guid_);
  }
  return OpBackend::AddNode(graph, stack);
}
} // namespace habana
