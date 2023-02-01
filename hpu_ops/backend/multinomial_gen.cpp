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

#include "generated/backend/multinomial.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/reduction_kernels.h"

namespace habana {

sizes_vec MultinomialOutputShape(const at::Stack& stack) {
  const torch::Tensor& t = stack_tensor(stack, 0);
  int64_t num_samples = stack.at(1).toInt();
  auto dim = t.sizes()[0];
  if (t.dim() == 1) {
    return {{num_samples}};
  }
  return {{dim, num_samples}};
}

std::shared_ptr<void> FillMultinomialParams(
    const at::Stack& stack,
    size_t& size) {
  at::ScalarType type = stack_tensor(stack, 0).scalar_type();
  float num_samples = stack.at(1).toInt();
  bool replacement = stack.at(2).toBool();
  PARAMS_STUB(ns_RandomMultinomial::ParamsV2);

  switch (type) {
    case at::ScalarType::Float:
    case at::ScalarType::BFloat16:
      params->num_samples = num_samples;
      params->replacement = replacement;
      break;
    default:
      TORCH_CHECK(false, "Unsupported type for random multinomial: ", type);
      break;
  }

  PT_KERNEL_DEBUG(
      __func__,
      " num_samples: ",
      params->num_samples,
      " replacement: ",
      params->replacement);

  return params;
}

void Multinomial::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  auto outshape = MultinomialOutputShape(stack)[0];
  auto params = FillMultinomialParams(stack, size);
  auto multinomial = BuildOp(
      graph,
      guid_,
      {syn_in(0), syn_in(1)},
      {{outshape, torch::kInt, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(multinomial[0]);
}
} // namespace habana
