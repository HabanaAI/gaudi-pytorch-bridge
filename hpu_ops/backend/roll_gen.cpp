/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "generated/backend/roll.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

FillParamsT FillRollParams(const at::Stack& stack) {
  PARAMS_STUB(ns_RollKernel::Params);

  constexpr int64_t shiftIndex = 1;
  constexpr int64_t dimsIndex = 2;

  auto shifts = stack.at(shiftIndex).toIntVector();
  auto dims = stack.at(dimsIndex).toIntVector();

  params->num_dims = safe_convert<unsigned int>(dims.size());

  for (size_t i = 0; i < shifts.size(); i++) {
    params->shifts[i] = safe_convert<int>(shifts[i]);
  }

  for (size_t i = 0; i < dims.size(); i++) {
    params->dims[i] = safe_convert<int>(dims[i]);
  }

  return paramsT;
}

SharedMetaDataVector RollSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  auto dtype = self.scalar_type();
  if (dtype == c10::ScalarType::Short) {
    dtype = c10::ScalarType::Int;
  }
  SharedMetaDataVector meta;

  if (!GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE)) {
    meta.reserve(1);
    auto& rollSharedMeta = meta.emplace_back("roll");
    rollSharedMeta.inputs_data.emplace_back(rank, dtype);
    rollSharedMeta.outputs_data.emplace_back(rank, dtype);
    return meta;
  }

  const auto splitRank = stack.at(2).toIntVector().empty() ? 1 : rank;
  SharedMetaTensor commonTensor = {splitRank, dtype};
  meta.reserve(2);

  auto& splitSharedMeta = meta.emplace_back("split");
  splitSharedMeta.inputs_data.push_back(commonTensor);
  splitSharedMeta.outputs_data = {commonTensor, commonTensor};

  auto& concatSharedMeta = meta.emplace_back("concat");
  concatSharedMeta.inputs_data = {commonTensor, commonTensor};
  concatSharedMeta.outputs_data.push_back(commonTensor);

  return meta;
}

void RollHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  constexpr int64_t inputIndex = 0;
  constexpr int64_t shiftIndex = 1;
  constexpr int64_t axisIndex = 2;

  auto input = stack.at(inputIndex).toTensor();

  // new i.e. cguid implementation for eager
  if (!GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE)) {
    auto params = FillRollParams(stack);

    auto result = BuildOp(
        graph,
        GetGuid(),
        {syn_in(0)},
        {{input.sizes(), ScalarType(), 0}},
        params.ptr(),
        params.size());

    syn_out(0) = std::move(result[0]);

    return;
  }

  // legacy implementation for lazy
  auto shift = stack.at(shiftIndex).toIntVector();
  auto axis = stack.at(axisIndex).toIntVector();

  // Handle the case for dims=None, SW-154508
  bool flatten_and_restore = axis.empty();

  auto original_input_shape = input.sizes();
  auto input_shape = original_input_shape;
  int64_t flattened_size = 0;

  HABANA_ASSERT(
      !shift.empty(), "roll: shift must be a scalar or a 1-D vector.");

  if (flatten_and_restore) {
    axis.push_back(0);
  }

  HABANA_ASSERT(!axis.empty(), "roll: axis must be a scalar or a 1-D vector.");
  HABANA_ASSERT(
      shift.size() == axis.size(),
      "roll: shift and axis must have the same size (",
      shift.size(),
      " != ",
      axis.size(),
      " ).");

  auto axisElementsCount = axis.size();

  auto* intermediate_input = syn_in(0);
  std::vector<synapse_helpers::tensor> intermediate_output;
  std::vector<synapse_helpers::tensor> reshape;

  if (flatten_and_restore) {
    flattened_size = input.numel();
    reshape.emplace_back(ReshapeHelper(
        graph, intermediate_input, {flattened_size}, ScalarType()));
    intermediate_input = reshape.at(0).get();
    input_shape = c10::ArrayRef<int64_t>(flattened_size);
  }

  unsigned int to_shift;
  unsigned int remain_shift;
  unsigned int mod_shift;

  // Iterate over the axis
  for (unsigned int i = 0; i < axisElementsCount; i++) {
    // Get axis and shift value
    auto shift_flat = shift[i];
    auto axis_flat = axis[i];

    // Handle negative axis
    axis_flat = c10::maybe_wrap_dim(axis_flat, input.dim(), true);
    auto axis_flat_size_t = safe_convert<size_t>(axis_flat);

    mod_shift = 0;
    if (input_shape[axis_flat_size_t] != 0) {
      mod_shift = safe_convert<unsigned int>(
          abs(shift_flat) % input_shape[axis_flat_size_t]);
    }

    // Handle when shift value > larger/smaller than shape of the input tensor
    if (shift_flat > 0) {
      to_shift = (shift_flat > input_shape[axis_flat_size_t])
          ? safe_convert<unsigned int>(
                input_shape[axis_flat_size_t] -
                safe_convert<int64_t>(mod_shift))
          : safe_convert<unsigned int>(
                input_shape[axis_flat_size_t] - shift_flat);
    } else {
      to_shift = (shift_flat < 0) ? mod_shift
                                  : safe_convert<unsigned int>(abs(shift_flat));
    }
    remain_shift = safe_convert<unsigned int>(
        input_shape[axis_flat_size_t] - safe_convert<int64_t>(to_shift));

    auto is_final_output = !flatten_and_restore && i == (axisElementsCount - 1)
        ? c10::make_optional<int>(0)
        : std::nullopt;

    if (to_shift != 0 && remain_shift != 0) {
      // Calculate the output shape
      auto out_shape_0 = input_shape.vec();
      auto out_shape_1 = input_shape.vec();
      out_shape_0[axis_flat_size_t] = safe_convert<int64_t>(to_shift);
      out_shape_1[axis_flat_size_t] = safe_convert<int64_t>(remain_shift);

      auto dim =
          safe_convert<unsigned>((input_shape.size() - 1) - axis_flat_size_t);

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

  if (flatten_and_restore) {
    syn_out(0) = ReshapeHelper(
        graph, intermediate_input, original_input_shape, ScalarType(), 0);
  } else {
    syn_out(0) = std::move(intermediate_output.at(0));
  }
}

} // namespace habana
