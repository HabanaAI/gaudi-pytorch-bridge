/******************************************************************************
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

#include "hpu_ops/rotary_embedding.h"

namespace habana {

RotaryEmbedding::RotaryEmbedding(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "rope_st2_fwd", scalar_type, {0}, {}, {}, false) {}

void RotaryEmbedding::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "RotaryEmbedding::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto sin = getNextInput<TensorsPair>(stackGetter);
  auto cos = getNextInput<TensorsPair>(stackGetter);
  auto offset = getNextInput<int>(stackGetter);

  std::string guid =
      get_guid_with_precision("rope_st2_fwd", input.pt_t.scalar_type());

  ns_RoPESt2::Params params{};
  params.offset = offset;

  std::vector<synTensor> inputs = {input.syn_t, sin.syn_t, cos.syn_t};

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {input.pt_t.sizes(), input.pt_t.scalar_type(), 0}};

  auto output = OpBackend::BuildNode(
      this, graph, {guid, inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

} // namespace habana

static const auto& RotaryEmbeddingKernelRegistry = habana::KernelRegistry().add(
    "hpu::rotary_embedding",
    KERNEL_FN_GLOBAL(habana::RotaryEmbedding));