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

std::vector<synapse_helpers::tensor> GeluCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    c10::optional<int> final_result_index = c10::nullopt) {
  return OpBackend::BuildNode(
      op,
      graph,
      {"gelu_fwd_" + habana_helpers::name_suffix_from_type(op->ScalarType()),
       std::move(input),
       {{outshape, op->ScalarType(), final_result_index},
        {outshape, op->ScalarType()}}});
}

void Gelu::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto Gelu = GeluCommonFunc(this, graph, {syn_in(0)}, outshape, 0);
  syn_out(0) = std::move(Gelu[0]);
}

void GeluBwd::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();

  auto Gelu = GeluCommonFunc(this, graph, {syn_in(1)}, outshape);

  auto Gelu_bwd = BuildOp(
      graph,
      "gelu_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {syn_in(0), syn_in(1), Gelu[1].get()},
      {{outshape, ScalarType(), 0}});

  syn_out(0) = std::move(Gelu_bwd[0]);
}
} // namespace habana
