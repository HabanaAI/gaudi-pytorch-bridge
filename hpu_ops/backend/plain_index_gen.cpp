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

#include <cstddef>

#include "generated/backend/index.h"
#include "generated/backend/plain_index.h"

namespace habana {

OutputMetaDataVector PlainIndexMeta(const at::Stack& stack) {
  // IndexMeta support also PlainIndexMeta
  // There are only 2 parameters passed in stack:
  // input tensor and indices
  return IndexMeta(stack);
}

static std::shared_ptr<void> FillPlainIndexParams(
    size_t num_index_tensors,
    size_t& size) {
  PARAMS_STUB(ns_IndexKernel::Params);
  assert(
      num_index_tensors <=
      sizeof(params->self_permute_dims) / sizeof(params->self_permute_dims[0]));
  // there is no advanced indexing
  for (size_t i = 0; i < num_index_tensors; ++i) {
    params->advanced_indexing_dims[i] = false;
    params->self_permute_dims[i] = 0;
  }
  params->num_index_tensors = num_index_tensors;
  return params;
}

void PlainIndexHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = IndexMeta(stack)[0];
  StackGetter stackGetter(this, stack, "IndexHabanaOperator::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto indices = stackGetter.getNextInput<std::vector<TensorsPair>>();
  size_t size = 0;
  auto params = FillPlainIndexParams(indices.size(), size);
  std::vector<synTensor> index_input{input.syn_t};
  for (auto const& index : indices)
    index_input.push_back(index.syn_t);

  auto result = BuildOp(
      graph,
      get_guid_with_precision("index", meta.dtype),
      std::move(index_input),
      {{meta.shape, meta.dtype, 0}},
      params.get(),
      size);

  syn_out(0) = std::move(result[0]);
}

} // namespace habana
