/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "generated/backend/scatter.h"

using namespace torch;

namespace habana {

SharedMetaDataVector ScatterSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& index = stack_tensor(stack, 2);
  auto dtype = self.scalar_type();
  auto rank = self.dim();

  if (stack.at(3).isTensor()) {
    const auto& updates = stack_tensor(stack, 3);
    SharedMetaData scatterSharedMeta{"scatter_fwd"};
    scatterSharedMeta.inputs_data = {
        {rank, dtype},
        {index.dim(), index.scalar_type()},
        {updates.dim(), dtype}};
    scatterSharedMeta.outputs_data.emplace_back(rank, dtype);
    return {scatterSharedMeta};
  } else {
    SharedMetaData scatterValueSharedMeta{"scatter_value_fwd"};
    scatterValueSharedMeta.inputs_data = {
        {rank, dtype}, {index.dim(), index.scalar_type()}};
    scatterValueSharedMeta.outputs_data.emplace_back(rank, dtype);
    return {scatterValueSharedMeta};
  }
}

SharedMetaDataVector ScatterReduceSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& index = stack_tensor(stack, 2);
  const auto dtype = self.scalar_type();
  const auto rank = self.dim();

  SharedMetaData scatterReduceSharedMeta{"scatter_reduce_fwd"};
  scatterReduceSharedMeta.inputs_data = {
      {rank, dtype}, {index.dim(), index.scalar_type()}, {rank, dtype}};
  scatterReduceSharedMeta.outputs_data.emplace_back(rank, dtype);

  if (rank > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(rank, dtype);
    return {scatterReduceSharedMeta, constantSharedMeta};
  }
  return {scatterReduceSharedMeta};
}

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
