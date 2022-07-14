/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/roll.h"
#include "hpu_op_helper.h"

namespace habana {
void RollHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr int64_t inputIndex = 0;
  constexpr int64_t shiftIndex = 1;
  constexpr int64_t axisIndex = 2;

  auto input = stack.at(inputIndex).toTensor();
  auto shift = stack.at(shiftIndex).toIntVector();
  auto axis = stack.at(axisIndex).toIntVector();

  auto input_shape = input.sizes();

  HABANA_ASSERT(
      shift.size() >= 1, "roll: shift must be a scalar or a 1-D vector.");
  HABANA_ASSERT(
      axis.size() >= 1, "roll: axis must be a scalar or a 1-D vector.");
  HABANA_ASSERT(
      shift.size() == axis.size(),
      "roll: shift and axis must have the same size (",
      shift.size(),
      " != ",
      axis.size(),
      " ).");

  auto axisElementsCount = axis.size();

  auto intermediate_input = syn_in(0);
  std::vector<synapse_helpers::tensor> intermediate_output;

  unsigned int to_shift, remain_shift, mod_shift;

  // Iterate over the axis
  for (unsigned int i = 0; i < axisElementsCount; i++) {
    // Get axis and shift value
    auto shift_flat = shift[i];
    auto axis_flat = axis[i];

    // Handle negative axis
    axis_flat = c10::maybe_wrap_dim(axis_flat, input.dim(), true);

    mod_shift = abs(shift_flat) % input_shape[axis_flat];

    // Handle when shift value > larger/smaller than shape of the input tensor
    if (shift_flat > 0) {
      to_shift = (shift_flat > input_shape[axis_flat])
          ? (input_shape[axis_flat] - (mod_shift))
          : (input_shape[axis_flat] - shift_flat);
    } else {
      to_shift = (shift_flat < 0) ? (mod_shift) : abs(shift_flat);
    }
    remain_shift = input_shape[axis_flat] - to_shift;

    auto is_final_output =
        i == axisElementsCount - 1 ? c10::make_optional<int>(0) : c10::nullopt;

    if (to_shift != 0 && remain_shift != 0) {
      // Calculate the output shape
      auto out_shape_0 = input_shape.vec(), out_shape_1 = input_shape.vec();
      out_shape_0[axis_flat] = to_shift;
      out_shape_1[axis_flat] = remain_shift;

      auto dim = static_cast<unsigned>((input_shape.size() - 1) - axis_flat);

      auto split_out = BuildOp(
          graph,
          "split",
          {intermediate_input},
          {
              {out_shape_0, ScalarType()},
              {out_shape_1, ScalarType()},
          },
          &dim,
          sizeof(dim));

      intermediate_output = BuildOp(
          graph,
          "concat",
          {split_out.at(1).get(), split_out.at(0).get()},
          {{input_shape, ScalarType(), is_final_output}},
          &dim,
          sizeof(dim));

    } else {
      intermediate_output = BuildOp(
          graph,
          "identity",
          {intermediate_input},
          {{input_shape, ScalarType(), is_final_output}});
    }
    intermediate_input = intermediate_output.at(0).get();
  }
  syn_out(0) = std::move(intermediate_output.at(0));
}

} // namespace habana
