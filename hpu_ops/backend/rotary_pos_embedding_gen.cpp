/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "generated/backend/rotary_pos_embedding.h"
#include "generated/backend/rotary_pos_embedding_backward.h"
#include "habana_helpers/logging.h"

namespace habana {

SharedMetaDataVector RotaryPosEmbeddingFwdBwdSharedMeta(
    const at::Stack& stack,
    const std::string& guid) {
  const auto& input = stack.at(0).toTensor();
  const auto& sin = stack.at(1).toTensor();
  const auto& cos = stack.at(2).toTensor();
  const auto& position_ids = stack.at(3).toOptional<at::Tensor>();
  const auto precision_type = input.scalar_type();

  SharedMetaData rotary_pos_embedding_shared_meta{guid};
  rotary_pos_embedding_shared_meta.inputs_data = {
      getSharedMetaFromTensor(input),
      {sin.dim(), precision_type},
      {cos.dim(), precision_type}};
  if (position_ids)
    rotary_pos_embedding_shared_meta.inputs_data.push_back(
        getSharedMetaFromOptionalTensor(position_ids));
  rotary_pos_embedding_shared_meta.outputs_data = {
      getSharedMetaFromTensor(input)};

  return {rotary_pos_embedding_shared_meta};
}

SharedMetaDataVector RotaryPosEmbeddingFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return RotaryPosEmbeddingFwdBwdSharedMeta(stack, "rotary_pos_embedding_fwd");
}

SharedMetaDataVector RotaryPosEmbeddingBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  return RotaryPosEmbeddingFwdBwdSharedMeta(stack, "rotary_pos_embedding_bwd");
}

void RotaryPosEmbedding::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "RotaryPosEmbedding::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto sin = stackGetter.getNextInput<TensorsPair>();
  auto cos = stackGetter.getNextInput<TensorsPair>();
  auto position_ids = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto offset = stackGetter.getNextInput<long>();
  auto mode = stackGetter.getNextInput<long>();

  ns_RoPESt2::ParamsV2 params{};
  HABANA_ASSERT(
      offset <= std::numeric_limits<unsigned int>::max(),
      "Offset value exceeds the maximum limit for int.");
  params.offset = static_cast<unsigned int>(offset);
  params.mode = static_cast<RotaryPosEmbeddingMode_t>(mode);

  std::vector<synTensor> inputs = {input.syn_t, sin.syn_t, cos.syn_t};
  if (position_ids) {
    inputs.push_back(position_ids.value().syn_t);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {input.pt_t.sizes(), input.pt_t.scalar_type(), 0}};

  auto output = OpBackend::BuildNode(
      this, graph, {guid_, inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

void RotaryPosEmbeddingBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "RotaryPosEmbeddingBackward::AddNode");
  auto grad_in = stackGetter.getNextInput<TensorsPair>();
  auto sin = stackGetter.getNextInput<TensorsPair>();
  auto cos = stackGetter.getNextInput<TensorsPair>();
  auto position_ids = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto offset = stackGetter.getNextInput<long>();
  auto mode = stackGetter.getNextInput<long>();

  ns_RoPESt2::ParamsV2 params{};
  HABANA_ASSERT(
      offset <= std::numeric_limits<unsigned int>::max(),
      "Offset value exceeds the maximum limit for int.");
  params.offset = static_cast<unsigned int>(offset);
  params.mode = static_cast<RotaryPosEmbeddingMode_t>(mode);

  std::vector<synTensor> inputs{grad_in.syn_t, sin.syn_t, cos.syn_t};
  if (position_ids) {
    inputs.push_back(position_ids.value().syn_t);
  }

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {grad_in.pt_t.sizes(), grad_in.pt_t.scalar_type(), 0}};

  auto grad_out = OpBackend::BuildNode(
      this, graph, {guid_, inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(grad_out[0]);
}

} // namespace habana
