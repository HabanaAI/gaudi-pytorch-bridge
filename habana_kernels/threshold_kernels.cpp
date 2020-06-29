/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/threshold_kernels.h"

using namespace torch;

void habana::ThresholdBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
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

  auto grad_input = at::empty(self.sizes(), self.options());
  AllocateSynapseOutput(graph, grad_input, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
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

  size_t device_id = self[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string nodeType =
      "relu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  habana::ThresholdBackwardOperator Op(device_id, nodeType);
  std::vector<c10::IValue> stack = {
      IValue(grad_output), IValue(self), IValue(threshold)};
  std::vector<const at::Tensor*> pt_inputs{&grad_output, &self};

  size_t key = habana_helpers::getRecipeKey(nodeType, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(self.sizes(), self.options());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs({output});
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    auto graph = habana_helpers::create_graph(device_id, nodeType);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

static auto registry = torch::RegisterOperators().op(
    torch::RegisterOperators::options()
        .schema(
            "aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor")
        .impl_unboxedOnlyKernel<
            decltype(threshold_backward_hpu),
            &threshold_backward_hpu>(DispatchKey::HABANATensorId)
        .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
