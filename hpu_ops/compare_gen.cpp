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

namespace habana {
sizes_vec HabanaOperatorHelper::CompareOutputShape(
    const torch::Tensor& self,
    const torch::Tensor& other) {
  return {at::infer_size(self.sizes(), other.sizes())};
}

void CompareHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    bool is_output_persistent) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);

  const at::ScalarType& result_dtype = at::result_type(self, other);
  if (self.scalar_type() == result_dtype and
      other.scalar_type() == result_dtype) {
    // return without type promotion as both inputs are of the same dtype
    return HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent);
  }

  int cast_index =
      self.scalar_type() == result_dtype and other.scalar_type() != result_dtype
      ? 1
      : 0;
  const std::string& cast_from = habana_helpers::name_suffix_from_type(
      stack_tensor(stack, cast_index).scalar_type());
  const std::string& cast_to =
      habana_helpers::name_suffix_from_type(result_dtype);
  // Insert cast on the input with lower dtype
  auto cast = BuildOp(
      graph,
      "cast_" + cast_from + "_to_" + cast_to,
      {syn_in(cast_index)},
      {{stack_tensor(stack, cast_index).sizes(), result_dtype}});
  const auto& cast_output = cast.at(0).get();

  // Extract guid without the dtype suffix
  const std::string& guid = guid_.substr(0, guid_.find_last_of('_') + 1);
  auto outshape = HabanaOperatorHelper::CompareOutputShape(self, other)[0];

  // Construct inputs for compare op considering the cast output
  std::vector<synTensor> syn_inputs;
  if (cast_index == 0) {
    syn_inputs = {cast_output, syn_in(1)};
  } else {
    syn_inputs = {syn_in(0), cast_output};
  }
  auto op = BuildOp(
      graph,
      guid + cast_to,
      syn_inputs,
      {{outshape, result_dtype, is_output_persistent, 0}});

  syn_out(0) = std::move(op.at(0));
}

} // namespace habana
