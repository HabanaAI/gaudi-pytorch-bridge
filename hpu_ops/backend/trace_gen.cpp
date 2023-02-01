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
#include "generated/backend/trace.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

sizes_vec TraceOutputShape(const at::Stack&) {
  return {{}};
}

void Trace::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto self = stack.at(0).toTensor();
  std::vector<int64_t> inputshape(self.dim());
  inputshape = self.sizes().vec();

  ns_MatrixDiag::Params matdiag_params{};
  matdiag_params.rows = inputshape[0];
  matdiag_params.cols = inputshape[1];
  matdiag_params.pad = 0;

  ns_Reduction::Params reduce_params{};
  reduce_params.reductionDimension = 0;

  if (inputshape[1] > inputshape[0]) {
    // diagonal on input
    auto diag = BuildOp(
        graph,
        "matrix_diag_part_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{inputshape[0], ScalarType()}},
        &matdiag_params,
        sizeof(matdiag_params));

    // sum on output of diagonal
    auto sum = BuildOp(
        graph,
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {diag[0].get()},
        {{1, ScalarType(), 0}},
        &reduce_params,
        sizeof(reduce_params));

    // output of sum is the output of this op
    syn_out(0) = std::move(sum[0]);
  } else {
    // diagonal on input
    auto diag = BuildOp(
        graph,
        "matrix_diag_part_fwd_" +
            habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0)},
        {{inputshape[1], ScalarType()}},
        &matdiag_params,
        sizeof(matdiag_params));

    // sum on output of diagonal
    auto sum = BuildOp(
        graph,
        "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {diag[0].get()},
        {{1, ScalarType(), 0}},
        &reduce_params,
        sizeof(reduce_params));

    // output of sum is the output of this op
    syn_out(0) = std::move(sum[0]);
  }
}
} // namespace habana
