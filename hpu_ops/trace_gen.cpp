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
#include "hpu_op_helper.h"

namespace habana {

template <>
LazyTrace<at::Tensor>::LazyTrace(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&, bool)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  auto x = inputs.at(0).toTensor();
  // In CPU trace op promotes all int dtype input to Long.
  // Setting the HPU output to be of dtype = Long, as the CPU output for int
  // dtype input is Long.
  if (x.scalar_type() == c10::ScalarType::Int)
    set_scalar_type(c10::ScalarType::Long);
}

template <>
at::Tensor LazyTrace<at::Tensor>::get_result_overrideable() {
  HABANA_ASSERT(false, "Shouldn't be reachable");
  return {};
}

sizes_vec TraceOutputShape(const at::Stack&, bool) {
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
