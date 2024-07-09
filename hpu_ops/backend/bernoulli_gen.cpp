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
