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
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
std::shared_ptr<void> FillBernoulliParams(size_t& size) {
  PARAMS_STUB(ns_RandomBernoulli::Params);
  return params;
}

void Bernoulli::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  // When p is a scalar, convert to a tensor since tpc kernel takes probability
  // as the first and only input
  auto outshape = stack_tensor(stack, 0).sizes();
  // For "outplace" and "out" variant, self tensor is the probability input
  int p_index = IsInplace() ? 1 : 0;
  // For "tensor_out" and "float_out" variant
  if (!IsInplace() &&
      c10::isFloatingType(stack.at(1).toTensor().scalar_type())) {
    p_index = 1;
  }
  int seed_index = p_index + 1;

  auto bcastOp = BuildOp(
      graph, "broadcast", {syn_in(p_index)}, {{outshape, ScalarType()}});
  size_t size = 0;
  auto params = FillBernoulliParams(size);

  auto dest_type = ScalarType();
  dest_type = ScalarType() == c10::ScalarType::Float ? c10::ScalarType::Int
                                                     : c10::ScalarType::Short;
  auto op = BuildOp(
      graph,
      guid_,
      {bcastOp.at(0).get(), syn_in(seed_index)},
      {{outshape, dest_type}},
      params.get(),
      size);
  auto castOp =
      CastHelper(graph, op.at(0).get(), outshape, dest_type, ScalarType(), 0);
  syn_out(0) = std::move(castOp);
}
} // namespace habana
