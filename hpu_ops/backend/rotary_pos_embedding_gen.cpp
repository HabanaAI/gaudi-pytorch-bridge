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

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& rotary_pos_embedding_shared_meta = meta.emplace_back(guid);
  rotary_pos_embedding_shared_meta.inputs_data = {
      getSharedMetaFromTensor(input),
      {sin.dim(), precision_type},
      {cos.dim(), precision_type}};
  if (position_ids) {
    rotary_pos_embedding_shared_meta.inputs_data.push_back(
        getSharedMetaFromOptionalTensor(position_ids));
  }
  rotary_pos_embedding_shared_meta.outputs_data = {
      getSharedMetaFromTensor(input)};

  return meta;
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

static std::vector<long int> CalcFinalSinShape(
    const std::vector<long int>& sin_shape,
    const std::optional<std::vector<long int>>& position_ids_shape,
    unsigned int offset) {
  std::vector<long int> final_sin_shape;

  if (position_ids_shape) {
    auto position_ids_shape_ = position_ids_shape.value();
    const auto position_shape_0 = position_ids_shape_[0];
    const auto position_shape_1 = position_ids_shape_[1];
    const auto sin_rank = sin_shape.size();

    const bool position_reshape_needed = position_shape_0 != 1;

    if (position_reshape_needed) {
      position_ids_shape_ = {1, position_shape_0 * position_shape_1};
    }

    const auto sin_last_dim = sin_shape.back();
    final_sin_shape = {position_ids_shape_.back(), sin_last_dim};

    if (position_reshape_needed) {
      final_sin_shape = {position_shape_0, position_shape_1, sin_last_dim};
    }

    final_sin_shape.insert(final_sin_shape.end() - (sin_rank - 2), 1);
  } else {
    final_sin_shape = sin_shape;
  }

  if (offset != 0) {
    final_sin_shape[final_sin_shape.size() - 1] -= offset;
  }
  return final_sin_shape;
}

static void CheckInputShapes(
    const std::vector<long int>& input_shape,
    const std::vector<long int>& sin_shape,
    const std::optional<std::vector<long int>>& position_ids_shape,
    unsigned int offset) {
  std::vector<long int> final_sin_shape =
      CalcFinalSinShape(sin_shape, position_ids_shape, offset);

  auto it_sin = final_sin_shape.rbegin();
  for (auto it_in = input_shape.rbegin(); it_in != (input_shape.rend() - 1);
       it_in++) {
    const auto dist = std::distance(input_shape.rbegin(), it_in);

    HABANA_ASSERT(
        (*it_sin == 1) || (*it_in == *it_sin),
        "Final sinus and cosinus tensor dim size final_sin_shape[",
        dist,
        "] = ",
        *it_sin,
        " differs from input dim size input_shape[",
        dist,
        "] = ",
        *it_in,
        ". They should be equal in case when final sin & cos dim size is not 1");
    it_sin++;
  }
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

  const auto input_shape = input.pt_t.sizes().vec();
  const auto sin_shape = sin.pt_t.sizes().vec();

  CheckInputShapes(
      input_shape,
      sin_shape,
      position_ids ? std::optional<std::vector<long int>>(
                         position_ids.value().pt_t.sizes().vec())
                   : std::nullopt,
      offset);

  if ((habana::ShapeInference::GetCurrentPass() ==
       habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
      graph.is_dry_run() && graph.is_dynamic_graph()) {
    const synapse_helpers::tensor& input_tensor = ReadSynInput(0);
    const synapse_helpers::tensor& sin_tensor = ReadSynInput(1);

    const auto input_current_shape =
        std::get<1>(habana::ShapeInference::GetMinMaxShape(input_tensor.id()));
    const auto sin_current_shape =
        std::get<1>(habana::ShapeInference::GetMinMaxShape(sin_tensor.id()));

    std::optional<std::vector<long int>> position_ids_current_shape_opt =
        std::nullopt;
    if (position_ids) {
      const synapse_helpers::tensor& pos_tensor = ReadSynInput(3);
      position_ids_current_shape_opt =
          std::get<1>(habana::ShapeInference::GetMinMaxShape(pos_tensor.id()));
    }

    if (!input_current_shape.empty() && !sin_current_shape.empty()) {
      CheckInputShapes(
          input_current_shape,
          sin_current_shape,
          position_ids_current_shape_opt,
          offset);
    }
  }

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

  const auto grad_in_shape = grad_in.pt_t.sizes().vec();
  const auto sin_shape = sin.pt_t.sizes().vec();

  CheckInputShapes(
      grad_in_shape,
      sin_shape,
      position_ids ? std::optional<std::vector<long int>>(
                         position_ids.value().pt_t.sizes().vec())
                   : std::nullopt,
      offset);

  if ((habana::ShapeInference::GetCurrentPass() ==
       habana::ShapeInfo::InferencePass::MAX_SHAPE) &&
      graph.is_dry_run() && graph.is_dynamic_graph()) {
    const synapse_helpers::tensor& grad_in_tensor = ReadSynInput(0);
    const synapse_helpers::tensor& sin_tensor = ReadSynInput(1);

    const auto grad_in_current_shape = std::get<1>(
        habana::ShapeInference::GetMinMaxShape(grad_in_tensor.id()));
    const auto sin_current_shape =
        std::get<1>(habana::ShapeInference::GetMinMaxShape(sin_tensor.id()));

    std::optional<std::vector<long int>> position_ids_current_shape =
        std::nullopt;
    if (position_ids) {
      const synapse_helpers::tensor& pos_tensor = ReadSynInput(3);
      position_ids_current_shape =
          std::get<1>(habana::ShapeInference::GetMinMaxShape(pos_tensor.id()));
    }

    if (!grad_in_current_shape.empty() && !sin_current_shape.empty()) {
      CheckInputShapes(
          grad_in_current_shape,
          sin_current_shape,
          position_ids_current_shape,
          offset);
    }
  }

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
