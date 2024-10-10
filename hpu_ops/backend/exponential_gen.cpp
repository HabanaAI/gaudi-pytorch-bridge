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

#include "generated/backend/exponential.h"
#include "habana_kernels/random_gen_kernels.h"

namespace habana {

OutputMetaDataVector ExponentialMeta(const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  OutputMetaData meta;
  meta.shape = self.sizes().vec();
  meta.dtype = self.scalar_type();
  return {meta};
}

SharedMetaDataVector ExponentialSharedMeta(const at::Stack& stack) {
  auto input = stack_tensor(stack, 0);
  auto dtype = input.scalar_type();
  auto rank = input.dim();
  auto seed = stack.at(2);

  SharedMetaData randomSharedMeta("random_exponential_fwd");
  randomSharedMeta.inputs_data.emplace_back(
      1, seed.isTensor() ? seed.toTensor().scalar_type() : at::ScalarType::Int);
  randomSharedMeta.outputs_data.emplace_back(rank, dtype);
  return {randomSharedMeta};
}

std::shared_ptr<void> FillExponentialParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_RandomExponential::Params);
  float lambd = stack.at(1).toScalar().toFloat();
  TORCH_CHECK(
      lambd >= 0.0,
      "exponential_ expects lambda >= 0.0, but found lambda=",
      lambd);
  params->beta = 1.0f / lambd;
  return params;
}

void ExponentialSeedTensorInput::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = ExponentialMeta(stack)[0];
  size_t size = 0;
  auto params = FillExponentialParams(stack, size);
  std::vector<synTensor> inputs;

  if (stack.at(2).isTensor())
    inputs.push_back(syn_in(1));
  else
    inputs.push_back(syn_seed());

  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto exponential = BuildOp(
      graph,
      get_guid_with_precision("random_exponential_fwd", meta.dtype),
      std::move(inputs),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(exponential[0]);
}
} // namespace habana
