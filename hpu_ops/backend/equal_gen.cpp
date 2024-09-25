/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/equal.h"

namespace habana {

std::shared_ptr<void> FillEqualParams(const at::Stack& stack, size_t& size) {
  PARAMS_STUB(ns_EqualPt::Params);
  auto self_sizes = stack_tensor(stack, 0).sizes();
  auto other_sizes = stack_tensor(stack, 1).sizes();
  params->forceFalse = self_sizes.size() != other_sizes.size();
  return params;
}

OutputMetaDataVector EqualMeta(const at::Stack&) {
  OutputMetaData meta;
  meta.shape = {};
  meta.dtype = c10::ScalarType::Bool;
  return {meta};
}

SharedMetaDataVector EqualSharedMeta(const at::Stack& stack) {
  const auto self = stack_tensor(stack, 0);
  const auto selfRank = self.dim();
  const auto other = stack_tensor(stack, 1);
  const auto otherRank = other.dim();

  if (selfRank != otherRank) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(1, c10::ScalarType::Bool);
    return {constantSharedMeta};
  } else {
    const auto computeDtype = self.scalar_type();
    auto rank = std::max(selfRank, otherRank);
    SharedMetaData equalSharedMeta{"equal_fwd"};
    SharedMetaTensor commonTensor = {rank, computeDtype};
    equalSharedMeta.inputs_data = {commonTensor, commonTensor};
    equalSharedMeta.outputs_data.emplace_back(rank, c10::ScalarType::Bool);

    SharedMetaData reduceProdMultiDimSharedMeta{"reduce_prod_multi_dim_fwd"};
    reduceProdMultiDimSharedMeta.inputs_data.emplace_back(
        rank, c10::ScalarType::Float);
    reduceProdMultiDimSharedMeta.outputs_data.emplace_back(
        1, c10::ScalarType::Float);

    return {equalSharedMeta, reduceProdMultiDimSharedMeta};
  }
}

void Equal::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Equal::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto other = stackGetter.getNextInput<TensorsPair>();

  size_t paramsSize = 0;
  auto params = FillParams(stack, paramsSize);
  const auto meta = OutputMeta(stack)[0];

  auto self_size = self.pt_t.sizes();
  auto other_size = other.pt_t.sizes();

  // DS not yet ready due to SW-202624
  if (!graph.is_dynamic_graph()) {
    auto equal = BuildOp(
        graph,
        get_guid_with_precision("equal_pt_fwd", ScalarType()),
        {self.syn_t, other.syn_t},
        {{meta.shape, meta.dtype, 0}},
        params.get(),
        paramsSize);

    syn_out(0) = std::move(equal[0]);
  } else if (self_size == other_size) {
    auto eq = BuildOp(
        graph,
        get_guid_with_precision("equal_fwd", ScalarType()),
        {self.syn_t, other.syn_t},
        {{self_size, meta.dtype}});

    // Although it seems we could skip reduction in the case of (1) input shape
    // and pass result of equal_fwd directly to the output we can't actually do
    // it. It would break eager shape agnostic flow as it changes topology when
    // JIT graph cache HIT occurs.
    size_t size = 0;
    PARAMS_STUB(ns_Reduction::ParamsV2);
    params->reductionDimensionMask = 0;
    params->keepDim = false;
    auto reduce_prod = BuildOp(
        graph,
        "reduce_prod_multi_dim_fwd_f32",
        {eq[0].get()},
        {{meta.shape, meta.dtype, 0}},
        params.get(),
        size);

    syn_out(0) = std::move(reduce_prod[0]);
  } else { // inputs with different shape
    auto false_tensor = ConstantHelper(graph, false, meta.dtype, 1, 0);

    syn_out(0) = std::move(false_tensor);
  }
}
} // namespace habana
