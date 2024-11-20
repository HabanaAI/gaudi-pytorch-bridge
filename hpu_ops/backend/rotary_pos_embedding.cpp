/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
  StackGetter stackGetter(this, stack, "RotaryPosEmbedding::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto sin = stackGetter.getNextInput<TensorsPair>();
  auto cos = stackGetter.getNextInput<TensorsPair>();
  auto position_ids = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto offset = stackGetter.getNextInput<int>();
  auto mode = stackGetter.getNextInput<int>();

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
    : OpBackend(
          device_id,
          "rotary_pos_embedding_bwd",
          scalar_type,
          {0},
          {},
          {},
          false) {}

void RotaryPosEmbeddingBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "RotaryPosEmbeddingBackward::AddNode");
  auto grad_in = stackGetter.getNextInput<TensorsPair>();
  auto sin = stackGetter.getNextInput<TensorsPair>();
  auto cos = stackGetter.getNextInput<TensorsPair>();
  auto position_ids = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto offset = stackGetter.getNextInput<int>();
  auto mode = stackGetter.getNextInput<int>();

  ns_RoPESt2::ParamsV2 params{};
  params.offset = offset;
  params.mode = static_cast<RotaryPosEmbeddingMode_t>(mode);

  std::vector<synTensor> inputs{grad_in.syn_t, sin.syn_t, cos.syn_t};
  if (position_ids) {
    inputs.push_back(position_ids.value().syn_t);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {grad_in.pt_t.sizes(), grad_in.pt_t.scalar_type(), 0}};

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
