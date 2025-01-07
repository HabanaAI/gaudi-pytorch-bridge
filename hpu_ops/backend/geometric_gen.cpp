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

#include "generated/backend/geometric.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
std::shared_ptr<void> FillRandomNegativeBinomialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomNegativeBinomial::ParamsV2);
  auto self = stack.at(0).toTensor();
  auto p = stack.at(1).toScalar().to<float>();

  params->p = p;
  params->k = 1.0;
  params->isAdditionEnable = true;

  return params;
}

SharedMetaDataVector GeometricSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);
  auto rank = self.dim();
  auto dtype = self.scalar_type();

  SharedMetaData geometricSharedMeta{"random_negative_binomial_fwd"};
  geometricSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
  geometricSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {geometricSharedMeta};
}

void Geometric::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  size_t size = 0;
  const auto& params = FillRandomNegativeBinomialParams(stack, size);

  auto geometric = BuildOp(
      graph,
      get_guid_with_precision("random_negative_binomial_fwd", ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);

  syn_out(0) = std::move(geometric[0]);
}
} // namespace habana
