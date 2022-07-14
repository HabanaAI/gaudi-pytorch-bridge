/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/fmod.h"

namespace habana {

void Fmod::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& result_type = ScalarType();

  if (result_type == at::kFloat or result_type == at::kBFloat16) {
    return OpBackend::AddNode(graph, stack);
  }
  // TPC supports only float and bfloat, for other types cast to float and
  // perform fmod and then, cast back to original type

  auto cast0 = CastHelper(
      graph,
      syn_in(0),
      stack.at(0).isTensor() ? stack_tensor(stack, 0).sizes() : 1,
      result_type,
      torch::kFloat);
  auto cast1 = CastHelper(
      graph,
      syn_in(1),
      stack.at(1).isTensor() ? stack_tensor(stack, 1).sizes() : 1,
      result_type,
      torch::kFloat);

  const auto outshape = BinaryOutputShape(stack, true)[0];
  auto fmod = BuildOp(
      graph,
      "mod_fwd_f32",
      {cast0.get(), cast1.get()},
      {{outshape, torch::kFloat}});

  auto final_cast =
      CastHelper(graph, fmod[0].get(), outshape, torch::kFloat, result_type, 0);
  syn_out(0) = std::move(final_cast);
}
} // namespace habana
