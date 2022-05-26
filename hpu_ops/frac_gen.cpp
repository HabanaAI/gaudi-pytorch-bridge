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
static auto BuildFrac(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    at::ScalarType dtype,
    at::IntArrayRef outshape,
    int out_index) {
  // sign on input 0
  auto sign = OpBackend::BuildNode(
      op,
      graph,
      {"sign_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {input},
       {{outshape, dtype}}});

  // abs on output of sign -> modulus
  auto abs_val = OpBackend::BuildNode(
      op,
      graph,
      {"abs_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {input},
       {{outshape, dtype}}});

  // floor on output of mod
  auto floor_val = OpBackend::BuildNode(
      op,
      graph,
      {"floor_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {abs_val[0].get()},
       {{outshape, dtype}}});

  // mul on output of floor & sign
  auto mul = OpBackend::BuildNode(
      op,
      graph,
      {MULT_GUID + habana_helpers::name_suffix_from_type(dtype),
       {floor_val[0].get(), sign[0].get()},
       {{outshape, dtype}}});

  // sub on input & output of mul
  return OpBackend::BuildNode(
      op,
      graph,
      {"sub_" + habana_helpers::name_suffix_from_type(dtype),
       {input, mul[0].get()},
       {{outshape, dtype, out_index}}});
}

void Frac::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto out = BuildFrac(this, graph, syn_in(0), ScalarType(), outshape, 0);
  syn_out(0) = std::move(out[0]);
}

void ForeachFrac::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildFrac(
        this, graph, syn_in(i), tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
