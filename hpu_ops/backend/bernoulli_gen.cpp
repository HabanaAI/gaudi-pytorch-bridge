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

#include "generated/backend/bernoulli.h"

namespace habana {
static auto bernoulli_impl(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor p,
    synTensor seed,
    at::IntArrayRef outshape,
    at::ScalarType dtype) {
  std::vector<synTensor> inputs = {};
  inputs.push_back(p);
  inputs.push_back(seed);
  op->CreateShapeTensorInput(graph, op->ScalarType(), outshape, inputs);

  auto bernoulli_out_dtype = dtype == c10::ScalarType::Float
      ? c10::ScalarType::Int
      : c10::ScalarType::Short;

  // Empty params with optional seed but still required to be filled to
  // bypass tpc kernel glue check
  PARAMS_STUB_VARS(ns_RandomBernoulli::Params, params, params_size);
  auto bernoulli = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("pt_bernoulli", dtype),
       inputs,
       {{outshape, dtype, 0}},
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
  auto p = syn_in(1); // ignore self when p is present
  auto seed = syn_in(2);
  syn_out(0) = std::move(
      bernoulli_impl(this, graph, p, seed, outshape, ScalarType())[0]);
}
} // namespace habana
