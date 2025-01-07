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

#include "generated/backend/exponential.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

OutputMetaDataVector ExponentialMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

SharedMetaDataVector ExponentialSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();
  auto seed = stack.at(2);

  SharedMetaData randomSharedMeta("random_exponential_fwd");
  randomSharedMeta.inputs_data.emplace_back(
      1, seed.isTensor() ? seed.toTensor().scalar_type() : at::ScalarType::Int);
  randomSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {randomSharedMeta};
}

std::shared_ptr<void> FillExponentialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = stack.at(1).toScalar().toFloat();
  TORCH_CHECK(
      lambd >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambd);
  params->beta = 1.0f / lambd;
  return params;
}

void ExponentialSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = ExponentialMeta(stack)[0];
  size_t size = 0;
  auto params = FillExponentialParams(stack, size);
  std::vector<synTensor> inputs;

  if (stack.at(2).isTensor())
    inputs.push_back(syn_in(1));
  else
    inputs.push_back(syn_seed());

  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto exponential = BuildOp(
      graph,
      get_guid_with_precision("random_exponential_fwd", meta.dtype),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(exponential[0]);
}
} // namespace habana
