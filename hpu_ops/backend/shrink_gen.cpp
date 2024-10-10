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
  PARAMS_STUB(ns_ShrinkKernel::ParamsV2);
  float lambda = stack.at(index_lambda).toScalar().to<float>();
  params->lambda = lambda;
  params->bias = 0;
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

SharedMetaDataVector HardShrinkFwdSharedMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto dtype = self.scalar_type();
  auto rank = self.dim();
  float lambda = stack.at(1).toScalar().to<float>();

  SharedMetaData hardShrinkFwdMeta{"memcpy"};
  hardShrinkFwdMeta.guid = lambda < 0.0 ? "memcpy" : "shrink_fwd";
  hardShrinkFwdMeta.inputs_data.emplace_back(rank, dtype);
  hardShrinkFwdMeta.outputs_data = hardShrinkFwdMeta.inputs_data;
  return {hardShrinkFwdMeta};
}

SharedMetaDataVector HardShrinkBwdSharedMeta(const at::Stack& stack) {
  auto grad = stack_tensor(stack, 0);
  auto self = stack_tensor(stack, 1);
  auto selfDtype = self.scalar_type();
  auto selfRank = self.dim();
  auto gradDType = grad.scalar_type();
  auto gradRank = grad.dim();
  float lambda = stack.at(2).toScalar().to<float>();

  SharedMetaData hardShrinkBwdMeta{"memcpy"};
  hardShrinkBwdMeta.guid = lambda < 0.0 ? "memcpy" : "shrink_bwd";
  hardShrinkBwdMeta.inputs_data = {
      {gradRank, gradDType}, {selfRank, selfDtype}};
  hardShrinkBwdMeta.outputs_data = {hardShrinkBwdMeta.inputs_data[1]};
  return {hardShrinkBwdMeta};
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
  ns_ShrinkKernel::ParamsV2 params{
      {lambda, 0.0}, -lambda, lambda, ShrinkMode_t::HARD_SHRINK};
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
  ns_ShrinkKernel::ParamsV2 params{
      {lambda, 0.0}, -lambda, lambda, ShrinkMode_t::HARD_SHRINK};
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
