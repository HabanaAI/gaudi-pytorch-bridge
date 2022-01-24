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
std::shared_ptr<void> FillExponentialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = stack.at(1).toScalar().toFloat();
  TORCH_CHECK(
      lambd >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambd);
  uint32_t seed = stack.at(2).to<uint32_t>();
  params->beta = 1.0 / lambd;
  params->seed = seed;
  return params;
}

void ExponentialIntSeedInput::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto outshape = stack_tensor(stack, 0).sizes();
  size_t size = 0;
  auto params = FillExponentialParams(stack, size);
  auto exponential = BuildOp(
      graph,
      "random_exponential_fwd_" +
          habana_helpers::name_suffix_from_type(ScalarType()),
      {},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}},
      params.get(),
      size);
  syn_out(0) = std::move(exponential[0]);
}
} // namespace habana
