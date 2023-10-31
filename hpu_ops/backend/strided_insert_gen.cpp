/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "habana_kernels/basic_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace sh = synapse_helpers;

namespace habana {

HPU_OP_BACKEND(_StridedInsert_Backend)

struct StridedInsert_Backend : _StridedInsert_Backend {
  StridedInsert_Backend(int device_id, c10::ScalarType scalar_type)
      : _StridedInsert_Backend(
            device_id,
            "strided_insert",
            scalar_type,
            {},
            {0},
            {},
            false) {}
};

void _StridedInsert_Backend::AddNode(sh::graph& graph, const at::Stack& stack) {
  size_t size = 0;
  PARAMS_STUB(synStridedOpParams);
  StridedInsertOperator::compute_params(*this, *params, stack, graph);

  auto num_syn_inputs = GetSynInputs().size();
  std::vector<synTensor> syn_inputs;
  syn_inputs.reserve(num_syn_inputs);
  for (size_t i = 0; i < num_syn_inputs; ++i) {
    syn_inputs.emplace_back(syn_in(i));
  }

  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto result = BuildOp(
      graph,
      guid_,
      std::move(syn_inputs),
      {{outshape, ScalarType(), 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
}

static const auto& kr_strided_insert = KernelRegistry().REGISTER_HPU_BACKEND(
    "hpu::strided_insert_",
    StridedInsert_Backend);

} // namespace habana