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
  auto bcastOp = OpBackend::BuildBroadcast(op, graph, p, outshape, dtype);

  auto bernoulli_out_dtype = dtype == c10::ScalarType::Float
      ? c10::ScalarType::Int
      : c10::ScalarType::Short;

  // Empty params with optional seed but still required to be filled to
  // bypass tpc kernel glue check
  PARAMS_STUB_VARS(ns_RandomBernoulli::Params, params, params_size);
  auto bernoulli = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("random_bernoulli_fwd", dtype),
       {bcastOp.get(), seed},
       {{outshape, bernoulli_out_dtype}},
       params.get(),
       params_size});

  return OpBackend::BuildCast(
      op,
      graph,
      bernoulli.at(0).get(),
      outshape,
      bernoulli_out_dtype,
      dtype,
      0);
}

void Bernoulli::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto p = syn_in(0); // self is p
  auto seed = syn_seed();
  syn_out(0) =
      std::move(bernoulli_impl(this, graph, p, seed, outshape, ScalarType()));
}

void BernoulliOut::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto p = syn_in(0); // self is p
  auto seed = syn_in(1);
  syn_out(0) =
      std::move(bernoulli_impl(this, graph, p, seed, outshape, ScalarType()));
}

void BernoulliWithP::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto p = syn_in(1); // ignore self when p is present
  auto seed = syn_in(2);
  syn_out(0) =
      std::move(bernoulli_impl(this, graph, p, seed, outshape, ScalarType()));
}

void BernoulliWithScalarP::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto outshape = stack_tensor(stack, 0).sizes();
  auto p = ConstantHelper(graph, stack.at(1).toScalar(), ScalarType());
  auto seed = syn_seed();
  syn_out(0) = std::move(
      bernoulli_impl(this, graph, p.get(), seed, outshape, ScalarType()));
}
} // namespace habana
