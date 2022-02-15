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

void ScatterOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack.at(0).toTensor();
  const auto index = stack.at(2).toTensor();
  const auto dim_ = stack.at(1).toInt();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  if (index.dim() == 0) {
    index.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }
  auto dim = at::maybe_wrap_dim(dim_, self.dim(), /*wrap_scalar=*/true);
  ns_ScatterKernel::Params params{};
  params.axis = static_cast<int>(self.dim() - dim - 1);
  bool bool_val = false;
  bool is_bool = false;
  at::Scalar val;
  if (!stack.at(3).isTensor()) {
    // Why do we need to do this?
    // We are converting the val to a real Bool (0 or 1) for src=Bool case.
    // If we don't do this, the kernel will execute, but the result from
    // TPC won't match CPU. This is because TPC doesn't have a true bool data
    // type and for cast we assume bool=i8. So, a val=4 will remain as 4 in the
    // final scattered output from TPC but for CPU the scattered result will
    // be 1. Hence the explicit conversion of val to bool.
    if (self.scalar_type() == c10::ScalarType::Bool) {
      is_bool = true;
      if (stack.at(3).isBool()) {
        bool_val = stack.at(3).toBool();
      } else if (stack.at(3).isInt()) {
        bool_val = stack.at(3).toInt() != 0;
      } else if (stack.at(3).isDouble()) {
        bool_val = static_cast<bool>(stack.at(3).toDouble());
      } else {
        bool_val = false;
      }
    }
  }

  auto src_or_val = stack.at(3).isTensor()
      ? std::make_unique<synapse_helpers::tensor>(
            std::move(p_context_->syn_inputs_.at(2).ref()))
      : std::make_unique<synapse_helpers::tensor>(ConstantHelper(
            graph,
            (is_bool ? bool_val : stack.at(3).toScalar()),
            ScalarType(),
            outshape));

  std::set<c10::ScalarType> int_types = {
      c10::ScalarType::Bool,
      c10::ScalarType::Char,
      c10::ScalarType::Byte,
      c10::ScalarType::Short};

  if ((self.scalar_type() != c10::ScalarType::Int) &&
      (int_types.find(self.scalar_type()) != int_types.end())) {
    // TPC scatter has support only for bf16, fp32 and i32
    auto cast_self =
        CastHelper(graph, syn_in(0), outshape, self.scalar_type(), torch::kInt);

    auto cast_src_or_val = CastHelper(
        graph,
        src_or_val->get(),
        stack.at(3).isTensor() ? stack_tensor(stack, 3).sizes() : outshape,
        self.scalar_type(),
        torch::kInt);

    std::vector<synTensor> syn_input_tensors = {
        cast_self.get(), syn_in(1), cast_src_or_val.get()};
    auto scatterkernel = BuildOp(
        graph,
        "scatter_fwd_" +
            habana_helpers::name_suffix_from_type(c10::ScalarType::Int),
        syn_input_tensors,
        {{outshape, c10::ScalarType::Int}},
        &params,
        sizeof(params));

    auto result_bool = CastHelper(
        graph,
        scatterkernel[0].get(),
        outshape,
        torch::kInt,
        self.scalar_type(),
        0);
    syn_out(0) = std::move(result_bool);
  } else {
    std::vector<synTensor> syn_input_tensors = {
        syn_in(0), syn_in(1), src_or_val->get()};
    auto scatterkernel = BuildOp(
        graph,
        "scatter_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        syn_input_tensors,
        {{outshape, ScalarType(), 0}},
        &params,
        sizeof(params));

    syn_out(0) = std::move(scatterkernel[0]);
  }
}
} // namespace habana
