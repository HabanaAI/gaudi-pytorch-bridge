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
sizes_vec ResizeOutputShape(const at::Stack& stack) {
  return {stack.at(1).toIntVector()};
}

void ResizeHabanaOperator::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto t = stack.at(0).toTensor();
  auto sizes = stack.at(1).toIntVector();
  auto memory_format = stack.at(2).toOptional<at::MemoryFormat>().value_or(
      t.suggest_memory_format());

  if (isOutputInfMode()) {
    GetOutputInfMeta().AddOutputTensor(TensorMetaData(
        sizes,
        CalculateStrides(sizes, memory_format),
        t.scalar_type(),
        memory_format));
    return;
  }

  p_context_->syn_outputs_.clear();
  p_context_->pt_outputs_.clear();

  // What if the same tensor is resized twice without getting flushed?

  OutputMetaData outMetaData;
  outMetaData.persistent = true;
  outMetaData.external = GetOutputMetaData(0).external; // temp WA for SW-156952
  const auto& output =
      habana::createPTTensor(t, sizes, t.options(), memory_format, true);
  AllocateSynapseOutput(graph, output, outMetaData);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

} // namespace habana
