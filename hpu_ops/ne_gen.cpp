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
void NE::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const at::Tensor self = stack_tensor(stack, 0);
  const auto& outshape = self.sizes();

  const at::ScalarType& result_type = c10::ScalarType::Bool;

  const at::ScalarType& input_dtype = stack.at(1).isScalar()
      ? self.scalar_type()
      : at::result_type(self, stack_tensor(stack, 1));

  std::vector<synTensor> syn_inputs;
  std::vector<synapse_helpers::tensor> cast;
  syn_inputs = {syn_in(0), syn_in(1)};

  if (!stack.at(1).isScalar()) {
    const at::Tensor other = stack_tensor(stack, 1);
    if ((self.scalar_type() != input_dtype and
         other.scalar_type() == input_dtype) ||
        (self.scalar_type() == input_dtype and
         other.scalar_type() != input_dtype)) {
      int cast_index = self.scalar_type() == input_dtype and
              other.scalar_type() != input_dtype
          ? 1
          : 0;
      const std::string& cast_from = habana_helpers::name_suffix_from_type(
          stack_tensor(stack, cast_index).scalar_type());
      const std::string& cast_to =
          habana_helpers::name_suffix_from_type(input_dtype);

      // Insert cast on the input with lower dtype
      cast = BuildOp(
          graph,
          "cast_" + cast_from + "_to_" + cast_to,
          {syn_in(cast_index)},
          {{stack_tensor(stack, cast_index).sizes(), input_dtype}});

      syn_inputs.at(cast_index) = cast[0].get();
    }
  }

  auto eq = BuildOp(
      graph,
      "equal_fwd_" + habana_helpers::name_suffix_from_type(input_dtype),
      syn_inputs,
      {{outshape, result_type, false}});

  // not on output of equal
  auto not_equal = BuildOp(
      graph,
      "not_fwd_i8",
      {eq[0].get()},
      {{outshape, result_type, is_output_persistent_list[0], true}});

  // output of not is the output of this op
  syn_out(0) = std::move(not_equal[0]);
}
} // namespace habana