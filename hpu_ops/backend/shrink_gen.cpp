/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/hardshrink.h"
#include "generated/backend/hardshrink_backward.h"
#include "generated/backend/softshrink.h"

namespace habana {
// mode_t = softshrink/hardshrink
// index_lambda  = index position of lambda
static std::shared_ptr<void> FillshrinkParams(
    const at::Stack& stack,
    size_t& size,
    ShrinkMode_t mode_t,
    int index_lambda) {
  PARAMS_STUB(ns_ShrinkKernel::TrainingParams);
  float lambda = stack.at(index_lambda).toScalar().to<float>();
  params->lowerBound = -lambda;
  params->upperBound = lambda;
  params->mode = mode_t;
  return params;
}

std::shared_ptr<void> FillsoftshrinkfwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillshrinkParams(stack, size, ShrinkMode_t::SOFT_SHRINK, 1);
}

std::shared_ptr<void> FillsoftshrinkbwdParams(
    const at::Stack& stack,
    size_t& size) {
  return FillshrinkParams(stack, size, ShrinkMode_t::SOFT_SHRINK, 2);
}

void HardShrinkFwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto dtype = ScalarType();
  const auto outshape = stack_tensor(stack, 0).sizes();

  int index_lambda = 1;
  float lambda = stack.at(index_lambda).toScalar().to<float>();

  if (lambda < 0.0) {
    auto out = OpBackend::BuildOp(
        graph, "memcpy", {syn_in(0)}, {{outshape, dtype, 0}});
    syn_out(0) = std::move(out[0]);
    return;
  }
  ns_ShrinkKernel::TrainingParams params{
      -lambda, lambda, ShrinkMode_t::HARD_SHRINK};
  auto out = OpBackend::BuildOp(
      graph,
      guid_,
      {syn_in(0)},
      {{outshape, dtype, 0}},
      &params,
      sizeof(params));
  syn_out(0) = std::move(out[0]);
  return;
}

void HardShrinkBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto dtype = ScalarType();
  const auto outshape = stack_tensor(stack, 1).sizes();

  int index_lambda = 2;
  float lambda = stack.at(index_lambda).toScalar().to<float>();

  if (lambda < 0.0) {
    auto out = OpBackend::BuildOp(
        graph, "memcpy", {syn_in(0)}, {{outshape, dtype, 0}});
    syn_out(0) = std::move(out[0]);
    return;
  }
  ns_ShrinkKernel::TrainingParams params{
      -lambda, lambda, ShrinkMode_t::HARD_SHRINK};
  auto out = OpBackend::BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {{outshape, dtype, 0}},
      &params,
      sizeof(params));
  syn_out(0) = std::move(out[0]);
  return;
}
} // namespace habana
