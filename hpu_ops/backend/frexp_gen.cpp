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

#include "generated/backend/frexp.h"

namespace habana {

OutputMetaDataVector FrexpMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  const auto shape = self.sizes().vec();

  OutputMetaData mantissaMeta, exponentMeta;
  mantissaMeta.shape = exponentMeta.shape = shape;

  mantissaMeta.dtype = self.scalar_type();
  exponentMeta.dtype = c10::ScalarType::Int;

  return {mantissaMeta, exponentMeta};
}

c10::ScalarType GetKernelExponentType(const c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::BFloat16:
    case c10::ScalarType::Half:
      return c10::ScalarType::Short;
    default:
      return c10::ScalarType::Int;
  }
}

SharedMetaDataVector FrexpSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto input = stack_tensor(stack, 0);
  auto inputType = input.scalar_type();
  auto rank = input.dim();

  if (c10::isIntegralType(inputType, true))
    inputType = c10::ScalarType::Float;

  auto exponentType = GetKernelExponentType(inputType);

  SharedMetaData frexpSharedMeta{"frexp"};
  frexpSharedMeta.inputs_data.emplace_back(rank, inputType);
  frexpSharedMeta.outputs_data = {{rank, exponentType}, {rank, inputType}};
  return {frexpSharedMeta};
}

void Frexp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = FrexpMeta(stack);
  const auto kernelExponentType = GetKernelExponentType(meta[0].dtype);
  const bool castIsNeededForExponent = kernelExponentType != meta[1].dtype;
  const c10::optional<int> exponentFinalResultIndex =
      castIsNeededForExponent ? c10::nullopt : c10::optional<int>{1};

  auto frexp = BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{meta[1].shape, kernelExponentType, exponentFinalResultIndex},
       {meta[0].shape, meta[0].dtype, 0}});

  syn_out(0) = std::move(frexp[1]);
  if (castIsNeededForExponent) {
    syn_out(1) = BuildCast(
        this,
        graph,
        frexp[0].get(),
        meta[1].shape,
        kernelExponentType,
        meta[1].dtype,
        1);
  } else {
    syn_out(1) = std::move(frexp[0]);
  }
}

} // namespace habana
