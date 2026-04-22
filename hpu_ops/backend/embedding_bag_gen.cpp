/**
 * Copyright (c) 2025-2026 Intel Corporation
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

#include <ATen/Context.h>
#include "generated/backend/_embedding_bag.h"
#include "generated/backend/_embedding_bag_backward.h"
#include "generated/backend/_embedding_bag_dense_backward.h"

namespace habana {

SharedMetaTensor getSharedMetaTensorFromTensorOnStack(
    const at::Stack& stack,
    int index) {
  const auto& v = stack.at(index);
  if (v.isTensor()) {
    const auto& t = v.toTensor();
    if (t.defined()) {
      return {t.dim(), t.scalar_type()};
    }
  }
  return createOptionalNotPresentSharedMetaTensor();
}

//  _embedding_bag_backward(
// 0) Tensor grad
// 1) Tensor indices
// 2) Tensor offsets
// 3) Tensor offset2bag
// 4) Tensor bag_size
// 5) Tensor maximum_indices
// 6) SymInt num_weights
// 7) bool scale_grad_by_freq
// 8) int mode
// 9) bool sparse
// 10) Tensor? per_sample_weights
// 11) int padding_idx=-1)
// OUT -> Tensor
SharedMetaDataVector EmbeddingBagBwdSharedMeta(
    const at::Stack& stack,
    [[maybe_unused]] habana_helpers::HabanaExecutionMode executionMode) {
  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& embeddingBagBwdSharedMeta = meta.emplace_back("embedding_bag_bwd");

  embeddingBagBwdSharedMeta.inputs_data.push_back(
      getSharedMetaTensorFromTensorOnStack(stack, 0));
  embeddingBagBwdSharedMeta.inputs_data.push_back(
      getSharedMetaTensorFromTensorOnStack(stack, 1));
  embeddingBagBwdSharedMeta.inputs_data.push_back(
      getSharedMetaTensorFromTensorOnStack(stack, 3));
  embeddingBagBwdSharedMeta.inputs_data.push_back(
      getSharedMetaTensorFromTensorOnStack(stack, 4));
  embeddingBagBwdSharedMeta.inputs_data.push_back(
      getSharedMetaTensorFromTensorOnStack(stack, 5));
  embeddingBagBwdSharedMeta.inputs_data.push_back(
      getSharedMetaTensorFromTensorOnStack(stack, 10));

  const auto& dtype = stack.at(0).toTensor().scalar_type();
  embeddingBagBwdSharedMeta.outputs_data.emplace_back(2, dtype);

  return meta;
}

bool EmbeddingBagBwdFallbackCheck(const bool sparse) {
  // HPU embedding_bag_bwd is non-deterministic due to parallel FP reductions in
  // IndexAddFwd op. Run on HPU only for dense + non-deterministic mode.
  return !sparse && !at::globalContext().deterministicAlgorithms();
}

bool EmbeddingBagDenseBwdFallbackCheck() {
  return !at::globalContext().deterministicAlgorithms();
}

FillParamsT FillEmbeddingBagDenseBwdParams(const at::Stack& stack) {
  PARAMS_STUB(ns_EmbeddingBagBwd::Params);
  const auto& num_weights = stack.at(5).toScalar().to<int64_t>();
  const auto& scale_grad_by_freq = stack.at(6).toScalar().to<bool>();
  const auto& mode = stack.at(7).toScalar().to<int64_t>();
  const bool has_per_sample_weights = stack.at(8).isTensor();
  const auto& padding_idx = stack.at(9).toScalar().to<int64_t>();

  params->num_weights = num_weights;
  params->scale_grad_by_freq = scale_grad_by_freq;
  params->mode = mode;
  params->has_per_sample_weights = has_per_sample_weights;
  params->padding_idx = padding_idx;

  return paramsT;
}

void EmbeddingBagBwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& params = FillEmbeddingBagBwdParams(stack);
  const auto meta = EmbeddingBagBwdMeta(stack)[0];

  std::vector<synTensor> inputs = {
      syn_in(0), syn_in(1), syn_in(3), syn_in(4), syn_in(5)};
  if (stack.at(10).isTensor()) {
    inputs.push_back(syn_in(6));
  }

  auto embedding_bag_bwd = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.ptr(),
       params.size()});
  syn_out(0) = std::move(embedding_bag_bwd.at(0));
}

FillParamsT FillEmbeddingBagBwdParams(const at::Stack& stack) {
  const auto& sparse = stack.at(9).toBool();
  HABANA_ASSERT(
      sparse == false, "Embedding Bag Sparse Bwd is not implemented.");

  PARAMS_STUB(ns_EmbeddingBagBwd::Params);
  const auto& num_weights = stack.at(6).to<int64_t>();
  const auto& scale_grad_by_freq = stack.at(7).toBool();
  const auto& mode = stack.at(8).to<int64_t>();
  const bool has_per_sample_weights = stack.at(10).isTensor();
  const auto& padding_idx = stack.at(11).to<int64_t>();

  params->num_weights = num_weights;
  params->scale_grad_by_freq = scale_grad_by_freq;
  params->mode = mode;
  params->has_per_sample_weights = has_per_sample_weights;
  params->padding_idx = padding_idx;

  return paramsT;
}

OutputMetaDataVector EmbeddingBagBwdMeta(const at::Stack& stack) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& num_weights = stack.at(6).to<int64_t>();
  const auto& sparse = stack.at(9).toBool();

  HABANA_ASSERT(
      sparse == false, "Embedding Bag Sparse Bwd is not implemented.");

  OutputMetaData meta;
  meta.dtype = grad.scalar_type();
  meta.shape.push_back(num_weights);
  meta.shape.push_back(grad.sizes()[1]);

  return {meta};
}

OutputMetaDataVector EmbeddingBagDenseBwdMeta(const at::Stack& stack) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& num_weights = stack.at(5).toScalar().to<int64_t>();

  OutputMetaData meta;
  meta.dtype = grad.scalar_type();
  meta.shape.push_back(num_weights);
  meta.shape.push_back(grad.sizes()[1]);

  return {meta};
}

OutputMetaDataVector EmbeddingBagMeta(const at::Stack& stack) {
  const auto& weight = stack_tensor(stack, 0);
  const auto& indices = stack_tensor(stack, 1);
  const auto& offsets = stack_tensor(stack, 2);
  const bool include_last_offset = stack.at(7).toScalar().toBool();

  const auto embedding_dim = weight.sizes()[1];
  const auto validOffsetsCount =
      offsets.sizes()[0] - (include_last_offset ? 1 : 0);
  const int64_t B =
      indices.sizes().size() == 1 ? validOffsetsCount : indices.sizes()[0];

  OutputMetaDataVector metaVec(4);

  auto& meta = metaVec[0];
  meta.dtype = weight.scalar_type();
  meta.shape = std::vector<int64_t>{B, embedding_dim};

  auto& meta_offset2bag = metaVec[1];
  meta_offset2bag.dtype = indices.scalar_type();
  meta_offset2bag.shape = std::vector<int64_t>{indices.sizes()[0]};

  auto& meta_bag_size = metaVec[2];
  meta_bag_size.dtype = indices.scalar_type();
  meta_bag_size.shape = std::vector<int64_t>{B};

  auto& max_meta = metaVec[3];
  max_meta.dtype = indices.scalar_type();
  max_meta.shape = std::vector<int64_t>{B, embedding_dim};

  return metaVec;
}

FillParamsT FillEmbeddingBagParams(const at::Stack& stack) {
  PARAMS_STUB(ns_EmbeddingBag::Params);
  bool scale_grad_by_freq = stack.at(3).toScalar().to<bool>();
  int64_t mode = stack.at(4).toScalar().to<int64_t>();
  bool sparse = stack.at(5).toScalar().to<bool>();
  auto has_per_sample_weights = stack.at(6).isTensor();
  bool include_last_offset = stack.at(7).toScalar().to<bool>();
  int64_t padding_idx = stack.at(8).toScalar().to<int64_t>();

  params->scale_grad_by_freq = scale_grad_by_freq;
  params->mode = mode;
  params->sparse = sparse;
  params->has_per_sample_weights = has_per_sample_weights;
  params->include_last_offset = include_last_offset;
  params->padding_idx = padding_idx;

  return paramsT;
}

} // namespace habana
