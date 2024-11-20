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

#include "generated/backend/ceil.h"
#include "generated/backend/floor.h"
#include "generated/backend/trunc.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
SharedMetaDataVector RoundingTruncSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return RoundingSharedMeta(stack, "trunc_fwd");
}

SharedMetaDataVector RoundingCeilSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return RoundingSharedMeta(stack, "ceil_fwd");
}

SharedMetaDataVector RoundingFloorSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  return RoundingSharedMeta(stack, "floor_fwd");
}

void RoundingFunc::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  std::string guid =
      c10::isIntegralType(ScalarType(), true) ? "identity" : guid_;
  auto result =
      BuildOp(graph, guid, {syn_in(0)}, {{outshape, ScalarType(), 0}});
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
