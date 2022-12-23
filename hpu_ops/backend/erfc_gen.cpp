/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "generated/backend/_foreach_erfc.h"
#include "generated/backend/erfc.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

static auto BuildErfc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor input,
    at::ScalarType dtype,
    at::IntArrayRef outshape,
    int out_index) {
  auto erf = OpBackend::BuildNode(
      op,
      graph,
      {"erf_fwd_" + habana_helpers::name_suffix_from_type(dtype),
       {input},
       {{outshape, dtype}}});

  auto constant = OpBackend::BuildConstant(op, graph, 1, dtype, outshape);

  return OpBackend::BuildNode(
      op,
      graph,
      {"sub_" + habana_helpers::name_suffix_from_type(dtype),
       {constant.get(), erf[0].get()},
       {{outshape, dtype, out_index}}});
}

void ForeachErfc::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    auto out = BuildErfc(
        this, graph, syn_in(i), tensor.scalar_type(), tensor.sizes(), i);
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
