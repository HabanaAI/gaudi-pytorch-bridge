/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/exponential.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {
std::shared_ptr<void> FillExponentialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = stack.at(1).toScalar().toFloat();
  TORCH_CHECK(
      lambd >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambd);
  params->beta = 1.0 / lambd;
  return params;
}

void ExponentialSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  // Discard self tensor, input is seed tensor only
  p_context_->syn_inputs_.pop_front();
  HABANA_ASSERT(p_context_->syn_inputs_.size() == 1);

  auto outshape = stack_tensor(stack, 0).sizes();
  size_t size = 0;
  auto params = FillExponentialParams(stack, size);
  auto exponential = BuildOp(
      graph,
      "random_exponential_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0)},
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(exponential[0]);
}
} // namespace habana
