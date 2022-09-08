/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/nansum.h"
#include "habana_kernels/reduction_kernels.h"
#include "reduction_template.h"

#define guidReducesum "reduce_sum_fwd_"

namespace habana {

template <>
LazyNansum<at::Tensor>::LazyNansum(
    const std::string& qualstring,
    const std::vector<at::IValue>& inputs,
    const std::function<sizes_vec(const at::Stack&)>& out_shapes_fn)
    : habana_lazy::LazyOp<at::Tensor>(qualstring, inputs, out_shapes_fn) {
  if (!inputs.at(3).isNone())
    set_scalar_type(inputs[3].toScalarType());
}

template <>
at::Tensor LazyNansum<at::Tensor>::get_result_overrideable() {
  throw std::runtime_error("Shouldn't be invoked");
}

sizes_vec NanSumIntListOutputShape(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);
  std::vector<int64_t> dim = stack.at(1).toIntList().vec();
  const bool keepdim = stack.at(2).toBool();
  std::vector<int64_t> shape =
      ReduceOperator::compute_output_shape(self, dim, keepdim);
  return {shape};
}

void NansumList::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto dtype = c10::ScalarType::Char;

  auto self = stack.at(0).toTensor();
  auto dim = stack.at(1).toIntVector();

  bool keepdim = stack.at(2).toBool();
  auto input = syn_in(0);
  auto input_in_dtype = HandleReductionDtype(
      this, graph, self, input, stack.at(3).toOptional<at::ScalarType>());
  if (input_in_dtype.has_value()) {
    input = input_in_dtype.value().get();
  }

  auto new_shape = NanSumIntListOutputShape(stack)[0];

  auto guid =
      guidReducesum + habana_helpers::name_suffix_from_type(ScalarType());

  // isNan on input
  auto is_nan = BuildOp(
      graph,
      "isnan_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {input},
      {{outshape, dtype}});

  auto zero_constant = ConstantHelper(graph, 0.0f, ScalarType(), outshape);

  // where on is_nan
  auto where = BuildOp(
      graph,
      "where_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {is_nan[0].get(), zero_constant.get(), input},
      {{outshape, ScalarType()}});

  auto reduce_sum = HandleReductionDimAndKeepdim(
      this,
      graph,
      self,
      {where[0].get()},
      dim,
      keepdim,
      guid,
      {{new_shape, ScalarType(), 0}});
  syn_out(0) = std::move(reduce_sum[0]);
}
} // namespace habana
