/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include "generated/backend/_foreach_erfc.h"
#include "generated/backend/erfc.h"

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
      {get_guid_with_precision("erf_fwd", dtype),
       {input},
       {{outshape, dtype}}});

  auto constant = OpBackend::BuildConstant(op, graph, 1, dtype, outshape);

  return OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub", dtype),
       {constant.get(), erf[0].get()},
       {{outshape, dtype, out_index}}});
}

void ForeachErfc::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& tensors = stack[0].toTensorList();
  for (auto i = 0u; i < tensors.size(); ++i) {
    const auto& tensor = tensors[i];
    const at::ScalarType scalar_type = tensor.scalar_type() != torch::kBFloat16
        ? torch::kFloat32
        : torch::kBFloat16;
    auto out =
        BuildErfc(this, graph, syn_in(i), scalar_type, tensor.sizes(), i);
    syn_out(i) = std::move(out[0]);
  }
}
} // namespace habana
