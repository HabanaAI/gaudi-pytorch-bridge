/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "backend/create_pt_tensor.h"
#include "generated/backend/resize.h"

namespace habana {
OutputMetaDataVector ResizeOutputMeta(const at::Stack& stack) {
  const torch::Tensor& self = stack_tensor(stack, 0);

  OutputMetaData meta;
  meta.shape = stack.at(1).toIntVector();
  meta.dtype = self.scalar_type();
  meta.persistent = true;
  return {meta};
}

void ResizeHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto t = stack.at(0).toTensor();
  auto meta = ResizeOutputMeta(stack)[0];
  auto memory_format = stack.at(2).toOptional<at::MemoryFormat>().value_or(
      t.suggest_memory_format());

  if (isOutputInfMode()) {
    GetOutputInfMeta().AddOutputTensor(TensorMetaData(
        meta.shape,
        CalculateStrides(meta.shape, memory_format),
        t.scalar_type(),
        memory_format));
    return;
  }

  p_context_->syn_outputs_.clear();
  p_context_->pt_outputs_.clear();

  // What if the same tensor is resized twice without getting flushed?

  const auto& output =
      habana::createPTTensor(t, meta.shape, t.options(), memory_format, true);
  AllocateSynapseOutput(graph, output, meta);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

} // namespace habana
