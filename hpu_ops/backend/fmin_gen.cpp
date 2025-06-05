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

#include "generated/backend/fmin.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {

void FMin::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  using namespace std::literals;
  const auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      std::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      std::nullopt,
      false,
      false);

  SetGuid(get_guid_with_precision(
      c10::isFloatingType(dtype) ? "fmin_fwd"sv : "min_fwd"sv, dtype));
  OpBackend::AddNode(graph, stack);
}

SharedMetaDataVector FMinSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      std::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      std::nullopt,
      false,
      false);

  const auto selfDim = stack.at(0).toTensor().dim();
  const auto otherDim = stack.at(1).toTensor().dim();

  SharedMetaData fMinMeta(c10::isFloatingType(dtype) ? "fmin_fwd" : "min_fwd");
  fMinMeta.inputs_data.emplace_back(selfDim, dtype);
  fMinMeta.inputs_data.emplace_back(otherDim, dtype);
  fMinMeta.outputs_data.emplace_back(std::max(selfDim, otherDim), dtype);
  return {fMinMeta};
}
} // namespace habana
