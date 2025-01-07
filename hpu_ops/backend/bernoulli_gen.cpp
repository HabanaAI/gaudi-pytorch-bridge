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

#include "generated/backend/bernoulli.h"
#include "hpu_ops/habana_random_ops.h"

namespace habana {
std::shared_ptr<void> FillBernoulliWithPParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::ParamsV2);
  if (stack.at(1).isScalar()) {
    params->probability = stack.at(1).toScalar().toFloat();
  }
  return params;
}

SharedMetaDataVector BernoulliSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& seed = stack.at(1);
  auto seedRank = 1;
  auto seedDtype = c10::ScalarType::Int;
  if (seed.isTensor()) {
    auto seedTensor = seed.toTensor();
    seedRank = seedTensor.dim();
    seedDtype = seedTensor.scalar_type();
  }

  const auto p = stack_tensor(stack, 0);
  auto pRank = p.dim();
  auto pDtype = p.scalar_type();

  SharedMetaData bernoulliSharedMeta{"pt_bernoulli"};
  bernoulliSharedMeta.inputs_data = {
      {pRank, pDtype}, {seedRank, seedDtype}, {1, c10::ScalarType::Int}};
  bernoulliSharedMeta.outputs_data.emplace_back(pRank, pDtype);

  return {bernoulliSharedMeta};
}

SharedMetaDataVector BernoulliWithPSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto self = stack_tensor(stack, 0);
  auto selfRank = self.dim();
  auto selfDtype = self.scalar_type();
  auto p = stack.at(1);
  auto seed = stack.at(2);
  c10::optional<at::Tensor> seedOptionalTensor = c10::nullopt;
  auto isSeedTensor = seed.isTensor();
  bool seedHasValue;
  if (isSeedTensor) {
    seedOptionalTensor = seed.toOptional<at::Tensor>();
    seedHasValue = seedOptionalTensor.has_value();
  } else {
    seedHasValue = seed.toOptional<at::Generator>().has_value();
  }

  // Precision type and output shape will be taken from self tensor, so even if
  // it is not passed, due to the specificty of SharedLayer the first input must
  // be provided so that the precision type match. "P" tensor will be
  // casted self's dtype
  SharedMetaData bernoulliSharedMeta{"pt_bernoulli"};
  if (p.isScalar())
    bernoulliSharedMeta.inputs_data.emplace_back(selfRank, selfDtype);
  else
    bernoulliSharedMeta.inputs_data.emplace_back(p.toTensor().dim(), selfDtype);

  if (seedHasValue) {
    auto seedRank = 1;
    auto seedDtype = c10::ScalarType::Int;
    if (isSeedTensor) {
      seedRank = seedOptionalTensor.value().dim();
      seedDtype = seedOptionalTensor.value().scalar_type();
    }
    bernoulliSharedMeta.inputs_data.emplace_back(seedRank, seedDtype);
  } else {
    bernoulliSharedMeta.inputs_data.push_back(
        createOptionalNotPresentSharedMetaTensor());
  }

  bernoulliSharedMeta.inputs_data.emplace_back(1, c10::ScalarType::Int);
  bernoulliSharedMeta.outputs_data.emplace_back(selfRank, selfDtype);

  return {bernoulliSharedMeta};
}

static auto bernoulli_impl(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor p,
    synTensor seed,
    at::IntArrayRef outshape,
    at::ScalarType dtype,
    int final_result_index = 0) {
  std::vector<synTensor> inputs = {};
  inputs.push_back(p);
  inputs.push_back(seed);
  op->CreateShapeTensorInput(graph, op->ScalarType(), outshape, inputs);

  // Empty params with optional seed but still required to be filled to
  // bypass tpc kernel glue check
  PARAMS_STUB_VARS(ns_RandomBernoulli::ParamsV2, params, params_size);
  auto bernoulli = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pt_bernoulli", dtype),
       inputs,
       {{outshape, dtype, final_result_index}},
       params.get(),
       params_size});
  return bernoulli;
}

void Bernoulli::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto p = syn_in(0); // self is p
  auto seed = stack[1].isTensor() ? syn_in(1) : syn_seed();
  syn_out(0) = std::move(
      bernoulli_impl(this, graph, p, seed, outshape, ScalarType())[0]);
}

void BernoulliOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto p = syn_in(0); // self is p
  auto seed = syn_in(1);
  syn_out(0) = std::move(
      bernoulli_impl(this, graph, p, seed, outshape, ScalarType())[0]);
}

void BernoulliWithP::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto dtype = ScalarType();
  if (stack.at(1).isScalar()) {
    size_t size = 0;
    auto params = FillParams(stack, size);
    std::vector<synTensor> inputs = {nullptr, syn_in(1)};
    CreateShapeTensorInput(graph, dtype, outshape, inputs);

    auto bernoulli = BuildOp(
        graph,
        get_guid_with_precision("pt_bernoulli", dtype),
        std::move(inputs),
        {{outshape, dtype, 0}},
        params.get(),
        size);
    syn_out(0) = std::move(bernoulli[0]);
  } else {
    auto p = syn_in(1); // ignore self when p is present
    auto seed = syn_in(2);
    syn_out(0) =
        std::move(bernoulli_impl(this, graph, p, seed, outshape, dtype)[0]);
  }
}

HabanaBernoulli::HabanaBernoulli(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_bernoulli",
          scalar_type,
          {1},
          {},
          {},
          false) {}

void HabanaBernoulli::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 1);

  syn_out(0) = std::move(bernoulli_impl(
      this,
      graph,
      syn_in(1),
      syn_in(0),
      input.sizes().vec(),
      input.scalar_type())[0]);
}

HabanaBernoulliCheckpoint::HabanaBernoulliCheckpoint(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "habana_bernoulli",
          scalar_type,
          {0, 1},
          {},
          {},
          false) {}

void HabanaBernoulliCheckpoint::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto seed =
      BuildOp(graph, "identity", {syn_in(0)}, {{{}, at::ScalarType::Int, 0}});
  syn_out(0) = std::move(seed[0]);

  const auto& input = stack_tensor(stack, 1);
  syn_out(1) = std::move(bernoulli_impl(
      this,
      graph,
      syn_in(1),
      syn_in(0),
      input.sizes().vec(),
      input.scalar_type(),
      1)[0]);
}
} // namespace habana

static const auto& HabanaRandomKernelRegistry =
    habana::KernelRegistry().REGISTER_RANDOM_OP(bernoulli, Bernoulli);
