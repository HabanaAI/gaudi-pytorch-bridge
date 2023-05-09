/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/equal.h"

namespace habana {

void Equal::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const at::Tensor other = stack_tensor(stack, 1);
  auto self_size = self.sizes();
  auto other_size = other.sizes();
  const at::ScalarType& result_type = c10::ScalarType::Bool;
  auto reshape_size = std::accumulate(
      std::begin(self_size),
      std::end(self_size),
      1,
      std::multiplies<int64_t>());
  std::vector<int64_t> reshape_outshape = {reshape_size};
  size_t size = 0;

  // inputs having same shape
  if (self_size == other_size) {
    auto eq = BuildOp(
        graph,
        "equal_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
        {syn_in(0), syn_in(1)},
        {{self_size, result_type}});

    auto cast_i8_to_f32 = CastHelper(
        graph, eq[0].get(), self_size, result_type, c10::ScalarType::Float);

    auto reshape = ReshapeHelper(
        graph, cast_i8_to_f32.get(), reshape_outshape, c10::ScalarType::Float);

    PARAMS_STUB(ns_Reduction::Params);
    params->reductionDimension = 0;
    auto reduce_prod = BuildOp(
        graph,
        "reduce_prod_fwd_f32",
        {reshape.get()},
        {{1, c10::ScalarType::Float}},
        params.get(),
        size);

    auto cast_f32_to_i8 = CastHelper(
        graph, reduce_prod[0].get(), 1, c10::ScalarType::Float, result_type, 0);

    syn_out(0) = std::move(cast_f32_to_i8);

  } else { // inputs with different shape
    auto false_tensor = ConstantHelper(graph, false, result_type, 1, 0);

    syn_out(0) = std::move(false_tensor);
  }
}
} // namespace habana
