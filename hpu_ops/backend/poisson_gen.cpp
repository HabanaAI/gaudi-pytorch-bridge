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

#include "generated/backend/poisson.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {
std::shared_ptr<void> FillPoissonParams(const at::Stack&, size_t& size) {
  PARAMS_STUB(ns_RandomPoisson::Params);
  params->lambda = 0.0;
  params->poissonFlavor = RandomPoissonFlavor_t::WITH_DIST;
  return params;
}

SharedMetaDataVector PoissonSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);
  auto selfDtype = self.scalar_type();
  auto rank = self.dim();
  auto seed = stack.at(1);
  SharedMetaTensor seedSharedTensor = {1, c10::ScalarType::Int};
  if (seed.isTensor()) {
    const auto seedTensor = seed.toTensor();
    seedSharedTensor = {seedTensor.dim(), seedTensor.scalar_type()};
  }

  SharedMetaData poissonSharedMeta{"random_poisson_fwd"};
  poissonSharedMeta.inputs_data.emplace_back(rank, selfDtype);
  poissonSharedMeta.inputs_data.push_back(seedSharedTensor);
  poissonSharedMeta.outputs_data.emplace_back(rank, selfDtype);
  return {poissonSharedMeta};
}

HabanaPoisson::HabanaPoisson(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "habana_poisson", scalar_type, {1}, {}, {}, false) {}

void HabanaPoisson::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input_tensor = stack_tensor(stack, 1);
  const auto& dtype = input_tensor.scalar_type();
  std::vector<synTensor> inputs = {syn_in(1), syn_in(0)};

  size_t params_size = 0;
  const auto& params = FillPoissonParams(stack, params_size);
  auto poisson = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("random_poisson_fwd", dtype),
       inputs,
       {{input_tensor.sizes().vec(), dtype, 0}},
       params.get(),
       params_size});

  syn_out(0) = std::move(poisson[0]);
}

HabanaPoissonCheckpoint::HabanaPoissonCheckpoint(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_poisson",
          scalar_type,
          {0, 1},
          {},
          {},
          false) {}

void HabanaPoissonCheckpoint::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto seed =
      BuildOp(graph, "identity", {syn_in(0)}, {{{}, at::ScalarType::Int, 0}});
  syn_out(0) = std::move(seed[0]);

  const auto& input_tensor = stack_tensor(stack, 1);
  const auto& dtype = input_tensor.scalar_type();
  std::vector<synTensor> inputs = {syn_in(1), syn_in(0)};

  size_t params_size = 0;
  const auto& params = FillPoissonParams(stack, params_size);
  auto poisson = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("random_poisson_fwd", dtype),
       inputs,
       {{input_tensor.sizes().vec(), dtype, 1}},
       params.get(),
       params_size});

  syn_out(1) = std::move(poisson[0]);
}
} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry().REGISTER_RANDOM_OP(poisson, Poisson);
