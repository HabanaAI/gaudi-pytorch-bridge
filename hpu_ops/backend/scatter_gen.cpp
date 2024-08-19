/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/scatter.h"

using namespace torch;

namespace habana {

void ScatterOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto index = stack.at(2).toTensor();
  const auto dim_ = stack.at(1).toInt();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);

  if (index.dim() == 0) {
    SET_SIZE_STRIDE_1D(index);
  }


  if (stack.at(3).isTensor()) {
    ns_ScatterKernel::ParamsReduce params{};
    params.axis = get_dim_in_tpc_order(dim, self.dim());

    auto scatterkernel = BuildOp(
        graph,
        get_guid_with_precision("scatter_fwd", ScalarType()),
        {syn_in(0), syn_in(1), syn_in(2)},
        {{outshape, ScalarType(), 0}},
        &params,
        sizeof(params));

    syn_out(0) = std::move(scatterkernel[0]);

  } else {
    at::Scalar val;
    at::IValue ival = stack.at(3);
    // Why do we need to do this?
    // We are converting the val to a real Bool (0 or 1) for src=Bool case.
    // If we don't do this, the kernel will execute, but the result from
    // TPC won't match CPU. This is because TPC doesn't have a true bool data
    // type and for cast we assume bool=i8. So, a val=4 will remain as 4 in the
    // final scattered output from TPC but for CPU the scattered result will
    // be 1. Hence the explicit conversion of val to bool.
    if (self.scalar_type() == c10::ScalarType::Bool) {
      if (ival.isBool()) {
        val = ival.toBool();
      } else if (ival.isInt()) {
        val = ival.toInt() != 0;
      } else if (ival.isDouble()) {
        val = static_cast<bool>(ival.toDouble());
      } else {
        val = false;
      }
    } else if (c10::isIntegralType(self.scalar_type(), false)) {
      val = ival.isDouble() ? static_cast<int>(ival.toDouble()) : ival.toInt();
    } else {
      val = ival.toScalar();
    }
    ns_ScatterValueKernel::Params params{};
    params.dim = dim;
    params.value = val.toDouble();
    auto scatterkernel = BuildOp(
        graph,
        get_guid_with_precision("scatter_value_fwd", ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}},
        &params,
        sizeof(params));

    syn_out(0) = std::move(scatterkernel[0]);
  }
}

void ScatterWithReduceOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto dim = stack.at(1).toInt();
  const auto index = stack.at(2).toTensor();
  const auto value = stack.at(3).toScalar();
  const auto reduce = stack.at(4).to<c10::string_view>();

  if (index.dim() == 0) {
    SET_SIZE_STRIDE_1D(index);
  }

  ScatterReduceMode_t mode = (reduce == "add")
      ? ScatterReduceMode_t::SCATTER_REDUCE_SUM
      : ScatterReduceMode_t::SCATTER_REDUCE_PROD;

  ns_ScatterReduceKernel::Params params{};
  params.dim = dim;
  params.include_self = true;
  params.mode = mode;

  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto broadcasted_value = ConstantHelper(graph, value, ScalarType(), outshape);

  std::vector<synTensor> syn_input_tensors = {
      syn_in(0), syn_in(1), broadcasted_value.get()};

  auto scatterkernel = BuildOp(
      graph,
      get_guid_with_precision("scatter_reduce_fwd", ScalarType()),
      std::move(syn_input_tensors),
      {{outshape, ScalarType(), 0}},
      &params,
      sizeof(params));

  syn_out(0) = std::move(scatterkernel[0]);
}

} // namespace habana
