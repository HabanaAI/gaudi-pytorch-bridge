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
  p_context_->syn_outputs_.clear();
  p_context_->pt_outputs_.clear();

  // What if the same tensor is resized twice without getting flushed?

  auto t = stack.at(0).toTensor();
  auto sizes = stack.at(1).toIntVector();
  auto memory_format_opt = stack.at(2).isNone()
      ? at::nullopt
      : at::make_optional(stack.at(2).toMemoryFormat());
  const auto& output = habana::createPTTensor(
      t, sizes, t.options(), memory_format_opt, IsOutputPersistent(0));
  AllocateSynapseOutput(graph, output, GetOutputMetaData(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

} // namespace habana
