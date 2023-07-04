/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/searchsorted.h"

namespace habana {
OutputMetaDataVector SearchSortedMeta(const at::Stack& stack) {
  std::vector<int64_t> outshape;
  if (stack.at(1).isTensor()) {
    auto self = stack_tensor(stack, 1);
    outshape = self.sizes().vec();
  } else {
    outshape = {1};
  }
  bool out_int32 = stack.at(2).toBool();

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = out_int32 ? torch::kInt32 : torch::kLong;
  return {meta};
}

std::shared_ptr<void> FillSearchSortedParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SearchSorted::Params);
  bool right = stack.at(3).toBool();
  if (stack.at(4).isString()) {
    right = stack.at(4).toStringView() == "right";
  }

  params->right = right;
  return params;
}

void SearchSorted::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto sorted_sequence = stack.at(0).toTensor();
  std::optional<synapse_helpers::tensor> sorted_sequence_syn_helper;
  synTensor sorted_sequence_syn_t;

  if (stack.at(5).isTensor()) {
    ns_GatherElementsKernel::Params gather_params;
    gather_params.axis = 0;
    auto gathered = BuildOp(
        graph,
        get_guid_with_precision(
            "gather_elements_fwd", sorted_sequence.scalar_type()),
        {syn_in(0), syn_in(2)},
        {{sorted_sequence.sizes().vec(), sorted_sequence.scalar_type()}},
        &gather_params,
        sizeof(gather_params));
    sorted_sequence_syn_helper = std::move(gathered[0]);
    sorted_sequence_syn_t = sorted_sequence_syn_helper->get();
  } else {
    sorted_sequence_syn_t = syn_in(0);
  }

  auto outputMeta = SearchSortedMeta(stack)[0];
  size_t params_size = sizeof(ns_SearchSorted::Params);
  auto params = FillParams(stack, params_size);
  auto result = BuildOp(
      graph,
      guid_,
      {sorted_sequence_syn_t, syn_in(1)},
      {{outputMeta.shape, outputMeta.dtype, 0}},
      params.get(),
      params_size);
  syn_out(0) = std::move(result[0]);
}
} // namespace habana