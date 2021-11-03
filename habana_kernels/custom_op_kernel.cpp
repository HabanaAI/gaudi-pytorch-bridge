/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "habana_kernels/custom_op_kernel.h"

namespace habana {

void CustomOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  std::vector<bool> is_outputs_persistent{is_output_persistent};
  AllocateAndAddSynapseNode(graph, inputs, is_outputs_persistent);
}

void CustomOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == op_desc_.getInputsSize(),
      "Incorrect size of inputs expected for CustomOperator: ",
      op_desc_.getSchemaName());

  TORCH_CHECK(
      inputs[0].isTensor(),
      "Currently custom op supprts first input as tensor type");

  TORCH_CHECK(
      op_desc_.getOutputsSize() == is_output_persistent.size(),
      "AllocateAndAddSynapseNode for multiple outputs count doesn't match, CustomOperator: ",
      op_desc_.getSchemaName());

  auto self = inputs[0].toTensor();

  auto outputs_desc = op_desc_.getOutputs();
  for (unsigned i = 0; i < op_desc_.getOutputsSize(); ++i) {
    std::vector<int64_t> result_sizes = self.sizes().vec();
    if (op_desc_.hasOutputShapeFunc(i)) {
      custom_op::compute_output_shape_function output_shape_func =
          op_desc_.getOutputShapeFunc(i);
      result_sizes = output_shape_func(inputs);
    }
    auto output = habana_helpers::createPTTensor(
        self,
        result_sizes,
        self.options().dtype(outputs_desc[i].dtype),
        self.suggest_memory_format(),
        is_output_persistent[i]);
    AllocateSynapseOutput(graph, output, is_output_persistent[i]);
  }

  std::shared_ptr<void> params = nullptr;
  size_t params_size = 0;
  if (op_desc_.hasUserParamsFunc()) {
    auto params_alloc_func = op_desc_.getUserParamsAllocFunc();
    params = params_alloc_func(inputs, params_size);
  }
  AddNodeToSynapseGraph(graph, params.get(), params_size);
}

} // namespace habana