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

#include "hpu_ops/rotary_pos_embedding.h"

namespace habana {

RotaryPosEmbedding::RotaryPosEmbedding(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "rotary_pos_embedding_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {}

void RotaryPosEmbedding::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "RotaryPosEmbedding::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto sin = getNextInput<TensorsPair>(stackGetter);
  auto cos = getNextInput<TensorsPair>(stackGetter);
  auto position_ids = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto offset = getNextInput<int>(stackGetter);
  auto mode = getNextInput<int>(stackGetter);

  ns_RoPESt2::ParamsV2 params{};
  params.offset = offset;
  params.mode = static_cast<RotaryPosEmbeddingMode_t>(mode);

  std::vector<synTensor> inputs = {input.syn_t, sin.syn_t, cos.syn_t};
  if (position_ids) {
    inputs.push_back(position_ids.value().syn_t);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {input.pt_t.sizes(), input.pt_t.scalar_type(), 0}};

  auto output = OpBackend::BuildNode(
      this, graph, {GetGuid(), inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

RotaryPosEmbeddingBackward::RotaryPosEmbeddingBackward(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(device_id, "rope_st2_bwd", scalar_type, {0}, {}, {}, false) {}

void RotaryPosEmbeddingBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "RotaryPosEmbeddingBackward::AddNode");
  auto grad_in = getNextInput<TensorsPair>(stackGetter);
  auto sin = getNextInput<TensorsPair>(stackGetter);
  auto cos = getNextInput<TensorsPair>(stackGetter);
  auto offset = getNextInput<int>(stackGetter);
  auto mode = getNextInput<int>(stackGetter);

  auto output_shape = grad_in.pt_t.sizes().vec();

  std::vector<synTensor> inputs{grad_in.syn_t};
  std::vector<synapse_helpers::tensor> maybe_sliced;

  if (offset > 0) {
    auto sliced_shape = sin.pt_t.sizes().vec();
    sliced_shape.back() = output_shape.back();

    synSliceParamsV2 slice_params{};
    slice_params.axes[0] = 0;
    slice_params.starts[0] = offset;
    slice_params.ends[0] = output_shape.back() + offset;
    slice_params.steps[0] = 1;

    auto sin_slice = BuildOp(
        graph,
        "slice",
        {sin.syn_t},
        {{sliced_shape, ScalarType()}},
        &slice_params,
        sizeof(slice_params));
    maybe_sliced.emplace_back(std::move(sin_slice[0]));
    inputs.push_back(maybe_sliced[0].get());

    auto cos_slice = BuildOp(
        graph,
        "slice",
        {cos.syn_t},
        {{sliced_shape, ScalarType()}},
        &slice_params,
        sizeof(slice_params));
    maybe_sliced.emplace_back(std::move(cos_slice[0]));
    inputs.push_back(maybe_sliced[1].get());
  } else {
    inputs.push_back(sin.syn_t);
    inputs.push_back(cos.syn_t);
  }

  ns_RoPESt2::ParamsV2 params{};
  params.offset = offset;
  params.mode = static_cast<RotaryPosEmbeddingMode_t>(mode);

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {output_shape, grad_in.pt_t.scalar_type(), 0}};

  auto grad_out = OpBackend::BuildNode(
      this, graph, {GetGuid(), inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(grad_out[0]);
}

} // namespace habana

static const auto& RotaryPosEmbeddingKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::rotary_pos_embedding",
            KERNEL_FN_GLOBAL(habana::RotaryPosEmbedding))
        .add(
            "hpu::rotary_pos_embedding_backward",
            KERNEL_FN_GLOBAL(habana::RotaryPosEmbeddingBackward));
