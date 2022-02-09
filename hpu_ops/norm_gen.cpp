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

sizes_vec NormOutputShape(const at::Stack&, bool) {
  return {{}};
}

void NormHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto p = stack.at(1).toScalar();
  const auto dtype = stack.at(2).toScalarType();
  auto outshape = self.sizes();
  auto n_dims = self.dim();

  synTensor input_tensor = syn_in(0);
  std::vector<synapse_helpers::tensor> cast;

  if (dtype != ScalarType()) {
    std::string cast_guid = "cast_" +
        habana_helpers::name_suffix_from_type(ScalarType()) + "_to_" +
        habana_helpers::name_suffix_from_type(dtype);
    cast = BuildOp(graph, cast_guid, {input_tensor}, {{outshape, dtype}});
    input_tensor = cast[0].get();
  }

  if (p.toFloat() == 2.0) {
    if (n_dims <= 1 || self.sizes()[0] == 1) {
      auto mul = BuildOp(
          graph,
          MULT_GUID + habana_helpers::name_suffix_from_type(dtype),
          {input_tensor, input_tensor},
          {{outshape, dtype}});

      std::vector<synTensor> reduction_inputs = {mul[0].get()};
      std::vector<synapse_helpers::tensor> reshape;

      if (n_dims > 1) {
        auto reshape_outshape = self.numel();
        reshape.emplace_back(
            ReshapeHelper(graph, reduction_inputs[0], reshape_outshape, dtype));
        reduction_inputs = {reshape[0].get()};
      }

      ns_Reduction::Params reduce_params{};
      reduce_params.reductionDimension = 0;
      auto sum = BuildOp(
          graph,
          "reduce_sum_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          reduction_inputs,
          {{1, dtype}},
          &reduce_params,
          sizeof(reduce_params));

      auto sqrt = BuildOp(
          graph,
          "sqrt_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          {sum[0].get()},
          {{1, dtype, 0}});

      syn_out(0) = std::move(sqrt[0]);

    } else {
      auto norm = BuildOp(
          graph,
          "frobenius_norm_fwd_" + habana_helpers::name_suffix_from_type(dtype),
          {input_tensor},
          {{1, dtype, 0}});

      syn_out(0) = std::move(norm[0]);
    }

  } else {
    auto reshape_outshape = self.numel();
    std::vector<synapse_helpers::tensor> reshape;
    if (n_dims > 1) {
      reshape.emplace_back(
          ReshapeHelper(graph, input_tensor, reshape_outshape, dtype));
      input_tensor = reshape[0].get();
    }

    ns_LpNormKernel::Params lpnorm_params{};
    lpnorm_params.p = p.toFloat();
    lpnorm_params.dim = 0;
    lpnorm_params.eps = 0;
    auto norm = BuildOp(
        graph,
        "lpnorm_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {input_tensor},
        {{reshape_outshape, dtype}, {reshape_outshape, dtype}},
        &lpnorm_params,
        sizeof(lpnorm_params));

    auto reciprocal = BuildOp(
        graph,
        "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(dtype),
        {norm[1].get()},
        {{reshape_outshape, dtype}});

    synSliceParams slice_params{};
    slice_params.axes[0] = 0;
    slice_params.starts[0] = 0;
    slice_params.ends[0] = 1;
    slice_params.steps[0] = 1;
    auto slice = BuildOp(
        graph,
        "slice_" + habana_helpers::name_suffix_from_type(dtype),
        {reciprocal[0].get()},
        {{1, dtype, 0}},
        &slice_params,
        sizeof(slice_params));

    syn_out(0) = std::move(slice[0]);
  }
}
} // namespace habana
