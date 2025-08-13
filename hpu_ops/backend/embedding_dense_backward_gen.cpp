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

#include "generated/backend/embedding.h"
#include "generated/backend/embedding_dense_backward.h"

namespace habana {
OutputMetaDataVector EmbeddingDenseBwdMeta(const at::Stack& stack) {
  const auto& grad = stack_tensor(stack, 0);
  int num_weights = stack.at(2).toScalar().to<int>();

  OutputMetaData meta;
  meta.dtype = grad.scalar_type();
  meta.shape.push_back(num_weights);
  meta.shape.push_back(grad.sizes().back());
  return {meta};
}

SharedMetaDataVector EmbeddingDenseBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto gradOut = stack_tensor(stack, 0);
  auto indices = stack_tensor(stack, 1);
  auto dtype = gradOut.scalar_type();
  auto gradRank = gradOut.dim();

  SharedMetaData embeddingDenseBwdSharedMeta{"embedding_dense_pt_bwd"};
  embeddingDenseBwdSharedMeta.inputs_data = {
      {gradRank, dtype}, {indices.dim(), indices.scalar_type()}};
  embeddingDenseBwdSharedMeta.outputs_data.emplace_back(2, dtype);

  return {embeddingDenseBwdSharedMeta};
}

FillParamsT FillEmbeddingDenseBackwardParams(const at::Stack& stack) {
  PARAMS_STUB(ns_EmbeddingDensePtBwdKernel::Params);
  int num_weights = stack.at(2).toScalar().to<int>();
  int padding_idx = stack.at(3).toScalar().to<int>();
  bool scale_grad_by_freq = stack.at(4).toBool();

  params->num_weights = num_weights;
  params->padding_idx = padding_idx;
  params->scaleGradByFreq = scale_grad_by_freq;
  return paramsT;
}

void EmbeddingDenseBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& params = FillEmbeddingDenseBackwardParams(stack);
  const auto meta = EmbeddingDenseBwdMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto embedding = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.ptr(),
       params.size()});
  syn_out(0) = std::move(embedding.at(0));
}
} // namespace habana
