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

static auto BuildLog10(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    at::ScalarType dtype,
    at::IntArrayRef outshape,
    int out_index) {
  // log on input 0
  auto log = OpBackend::BuildNode(
      op,
      graph,
      {"log_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {input},
       {{outshape, dtype}}});

  // 1/ln(10) = 0.4342944819
  constexpr float value = 0.4342944819;
  auto constant_value = OpBackend::BuildConstant(op, graph, value, dtype);

  // mul on log of input and constant value
  return OpBackend::BuildNode(
      op,
      graph,
      {MULT_GUID + habana_helpers::name_suffix_from_type(dtype),
       {constant_value.get(), log[0].get()},
       {{outshape, dtype, out_index}}});
}
void Log10::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto output = BuildLog10(this, graph, syn_in(0), ScalarType(), outshape, 0);
  syn_out(0) = std::move(output[0]);
}

void ForeachLog10::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildLog10(
        this, graph, syn_in(i), tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
