/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/threshold_kernels.h"

using namespace torch;

void habana::ThresholdBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for threshold operator");

  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg3 type expected to be scalar");

  auto grad_output = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto threshold = inputs[2].toScalar();

  TORCH_CHECK(
      threshold.to<float>() == 0.0,
      "Threshold values other than 0 are not supported")

  auto grad_input =
      habana::createPTTensor(self, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, grad_input, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

habana::OutputShapeInfRetType habana::ThresholdBackwardOperator::
    ComputeOutputShape(torch::jit::Stack& inputs) {
  auto self = inputs[1].toTensor();
  habana::OutputShapeInfRetType out;
  out.AddOutputTensor(habana::TensorMetaData(
      self.sizes().vec(),
      HabanaOperator::CalculateStrides(
          self.sizes().vec(), self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

/***************************************************************************
 * @brief Implements backward pass for torch.nn.Threshold(threshold: float,
 *value: float)
 * @param grad_output: Input tensor for backward pass
 * @param self: Input tensor for forward pass
 * @param threshold:The value to threshold at
 ****************************************************************************/
Tensor threshold_backward_hpu(
    const Tensor& grad_output,
    const Tensor& self,
    Scalar threshold) {
  PT_KERNEL_BEGIN;

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string nodeType =
      "relu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  habana::ThresholdBackwardOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(grad_output), IValue(self), IValue(threshold)};
  std::vector<at::Tensor> pt_inputs{grad_output, self};

  size_t key = Op.GetRecipeKey(nodeType, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    std::vector<at::Tensor> v{output};
    Op.Execute(key, pt_inputs, v);
  } else {
    habana::OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}
