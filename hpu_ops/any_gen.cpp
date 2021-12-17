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

void Any::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  auto self = stack.at(0).toTensor();
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto cast_f32 = CastHelper(
      graph, syn_in(0), outshape, self.scalar_type(), c10::ScalarType::Float);

  auto self_size = self.sizes();
  auto reshape_size = std::accumulate(
      std::begin(self_size),
      std::end(self_size),
      1,
      std::multiplies<int64_t>());
  std::vector<int64_t> reshape_outshape = {reshape_size};

  auto reshape = BuildOp(
      graph,
      "reshape",
      {cast_f32.get()},
      {{reshape_outshape, c10::ScalarType::Float}});

  auto abs = BuildOp(
      graph,
      "abs_fwd_f32",
      {reshape[0].get()},
      {{reshape_outshape, c10::ScalarType::Float}});

  size_t size = 0;
  PARAMS_STUB(ns_Reduction::Params);
  auto reduce_sum = BuildOp(
      graph,
      "reduce_sum_fwd_f32",
      {abs[0].get()},
      {{1, c10::ScalarType::Float}},
      params.get(),
      size);

  constexpr float value = 0;
  auto constant_value = ConstantHelper(graph, value);
  auto out_shape = AllOutputShape(stack, true)[0];

  auto greater_than_zero = BuildOp(
      graph,
      "greater_fwd_f32",
      {reduce_sum[0].get(), constant_value.get()},
      {{out_shape, c10::ScalarType::Bool, is_output_persistent_list[0], 0}});

  syn_out(0) = std::move(greater_than_zero[0]);
}
} // namespace habana
