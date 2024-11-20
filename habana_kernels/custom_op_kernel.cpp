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

#include "habana_kernels/custom_op_kernel.h"
#include "backend/create_pt_tensor.h"

namespace habana {

void CustomOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == op_desc_.getInputsSize(),
      "Incorrect size of inputs expected for CustomOperator: ",
      op_desc_.getSchemaName());

  TORCH_CHECK(
      inputs[0].isTensor(),
      "Currently custom op supprts first input as tensor type");

  TORCH_CHECK(
      op_desc_.getOutputsSize() == output_metadata.size(),
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
    auto output = habana::createPTTensor(
        self,
        result_sizes,
        self.options().dtype(outputs_desc[i].dtype),
        self.suggest_memory_format(),
        output_metadata.at(i).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(i));
  }

  std::shared_ptr<void> params = nullptr;
  size_t params_size = 0;
  if (op_desc_.hasUserParamsFunc()) {
    auto params_alloc_func = op_desc_.getUserParamsAllocFunc();
    params = params_alloc_func(inputs, params_size);
  }
  AddNodeToSynapseGraph(graph, params.get(), params_size);
}

InferOutputMetaRetType CustomOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto self = inputs[0].toTensor();
  auto outputs_desc = op_desc_.getOutputs();
  for (unsigned i = 0; i < op_desc_.getOutputsSize(); ++i) {
    std::vector<int64_t> result_sizes = self.sizes().vec();
    if (op_desc_.hasOutputShapeFunc(i)) {
      custom_op::compute_output_shape_function output_shape_func =
          op_desc_.getOutputShapeFunc(i);
      result_sizes = output_shape_func(inputs);
    }
    out.AddOutputTensor(TensorMetaData(
        result_sizes,
        HabanaOperator::CalculateStrides(
            result_sizes, self.suggest_memory_format()),
        outputs_desc[i].dtype,
        self.suggest_memory_format()));
  }
  return out;
}

} // namespace habana
