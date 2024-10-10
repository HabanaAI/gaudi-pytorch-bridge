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

void Equal::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "Equal::AddNode");
  auto self = getNextInput<TensorsPair>(stackGetter);
  auto other = getNextInput<TensorsPair>(stackGetter);

  auto self_size = self.pt_t.sizes();
  auto other_size = other.pt_t.sizes();
  at::ScalarType result_type = c10::ScalarType::Bool;

  if (self_size == other_size) {
    auto eq = BuildOp(
        graph,
        get_guid_with_precision("equal_fwd", ScalarType()),
        {self.syn_t, other.syn_t},
        {{self_size, result_type}});

    int64_t reduced_dim = 1;
    c10::IntArrayRef reduced_size = reduced_dim;

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
        {{reduced_size, result_type, 0}},
        params.get(),
        size);

    syn_out(0) = std::move(reduce_prod[0]);
  } else { // inputs with different shape
    auto false_tensor = ConstantHelper(graph, false, result_type, 1, 0);

    syn_out(0) = std::move(false_tensor);
  }
}
} // namespace habana
