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
