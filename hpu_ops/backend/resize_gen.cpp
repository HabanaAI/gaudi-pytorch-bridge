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
#include "backend/create_pt_tensor.h"
#include "generated/backend/_resize_output.h"
#include "generated/backend/resize.h"

namespace habana {

namespace {

void resizeTensor(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    const at::Tensor& tensor,
    at::MemoryFormat memory_format) {
  auto meta = ResizeOutputMeta(stack)[0];

  if (op->isOutputInfMode()) {
    op->GetOutputInfMeta().AddOutputTensor(TensorMetaData(
        meta.shape,
        op->CalculateStrides(meta.shape, memory_format),
        tensor.scalar_type(),
        memory_format));
  } else {
    op->GetSynOutputs().clear();
    op->GetOutputs().clear();

    // What if the same tensor is resized twice without getting flushed?

    const auto& output = habana::createPTTensor(
        tensor, meta.shape, tensor.options(), memory_format, true);
    op->AllocateSynapseOutput(graph, output, meta);
  }
}

} // namespace

OutputMetaDataVector ResizeOutputMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.dtype = stack.at(0).toTensor().scalar_type();
  meta.shape = stack.at(1).toIntVector();
  meta.persistent = true;
  return {meta};
}

void ResizeOpBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto memory_format =
      stack.at(2).toOptional<at::MemoryFormat>().value_or(
          self.suggest_memory_format());

  resizeTensor(this, graph, stack, self, memory_format);
  if (isOutputInfMode()) {
    GetOutputInfMeta().AddNodeParams(nullptr, 0);
  }
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void ResizeOutputOpBackend::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& self = stack.at(0).toTensor();
  const auto device = stack.at(2).toDevice();

  TORCH_CHECK(
      self.device() == device, "Tensor doesn't have the correct device set");

  resizeTensor(this, graph, stack, self, self.suggest_memory_format());
  if (isOutputInfMode()) {
    GetOutputInfMeta().AddNodeParams(nullptr, 0);
  }
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

} // namespace habana
