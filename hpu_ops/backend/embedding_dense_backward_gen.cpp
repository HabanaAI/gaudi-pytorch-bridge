/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/embedding.h"
#include "generated/backend/embedding_dense_backward.h"

namespace habana {
OutputMetaDataVector EmbeddingDenseBwdMeta(const at::Stack& stack) {
  const auto& grad = stack_tensor(stack, 0);
  int num_weights = stack.at(2).toScalar().to<int>();

  OutputMetaData meta;
  meta.dtype = grad.scalar_type();
  meta.shape.push_back(num_weights);
  meta.shape.push_back(grad.sizes().vec().back());
  return {meta};
}

std::shared_ptr<void> FillEmbeddingDenseBackwardParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_EmbeddingDensePtBwdKernel::Params);
  int num_weights = stack.at(2).toScalar().to<int>();
  int padding_idx = stack.at(3).toScalar().to<int>();
  bool scale_grad_by_freq = stack.at(4).toBool();

  params->num_weights = num_weights;
  params->padding_idx = padding_idx;
  params->scaleGradByFreq = scale_grad_by_freq;
  return params;
}

void EmbeddingDenseBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  const auto& params = FillEmbeddingDenseBackwardParams(stack, size);
  const auto meta = EmbeddingDenseBwdMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};
  CreateShapeTensorInput(graph, meta.dtype, meta.shape, inputs);
  auto embedding = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});
  syn_out(0) = std::move(embedding.at(0));
}
} // namespace habana
