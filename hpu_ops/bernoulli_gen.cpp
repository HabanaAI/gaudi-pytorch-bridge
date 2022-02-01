
/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
std::shared_ptr<void> FillBernoulliParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::Params);
  params->seed = stack.back().toInt();
  return params;
}

std::shared_ptr<void> FillBernoulliOutParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::Params);
  params->seed = stack.at(1).toInt();
  return params;
}

void Bernoulli::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  // When p is a scalar, convert to a tensor since tpc kernel takes probability
  // as the first and only input
  auto outshape = stack_tensor(stack, 0).sizes();
  // For "outplace" and "out" variant, self tensor is the probability input
  int p_index = IsInplace() ? 1 : 0;
  auto p = stack.at(p_index).isTensor()
      ? std::make_unique<synapse_helpers::tensor>(
            std::move(p_context_->syn_inputs_.at(p_index).ref()))
      : std::make_unique<synapse_helpers::tensor>(ConstantHelper(
            graph, stack.at(1).toDouble(), ScalarType(), outshape));

  size_t size = 0;
  auto params = FillParams(stack, size);
  auto op = BuildOp(
      graph,
      guid_,
      {p->get()},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(op[0]);
}

} // namespace habana
