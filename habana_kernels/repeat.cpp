/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "repeat.h"
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;
using namespace habana;

std::vector<int64_t> RepeatOperator::compute_output_shape(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  std::vector<int64_t> outshape(repeats.size());
  for (size_t i = 0; i < repeats.size(); ++i) {
    outshape[i] = padded_size[i] * repeats[i];
  }
  return outshape;
}

std::vector<int64_t> RepeatOperator::compute_reshape_output(
    const at::Tensor& self,
    at::IntArrayRef repeats) {
  int64_t num_new_dimensions = repeats.size() - self.dim();
  std::vector<int64_t> padded_size(num_new_dimensions, 1);
  padded_size.insert(
      padded_size.end(), self.sizes().begin(), self.sizes().end());
  return padded_size;
}

void RepeatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for repeat operator");
  TORCH_CHECK(
      inputs[1].isIntList() || inputs[1].isTensor(),
      "Input arg2 expected to be intlist or tenspr shape for repeat operator");
  auto input = inputs[0].toTensor();
  auto repeats = inputs[1].isIntList() ? inputs[1].toIntVector()
                                       : inputs[1].toTensor().sizes().vec();
  int64_t size = repeats.size();

  if (size > input.ndimension()) {
    torch::jit::Stack temp_stack;
    auto reshapeSize = RepeatOperator::compute_reshape_output(input, repeats);
    auto reshapeOp = make_operator<ReshapeOperator>(
        this->p_context_->device_id_, input.scalar_type());
    temp_stack = {IValue(input), IValue(reshapeSize)};
    reshapeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    reshapeOp->AllocateAndAddSynapseNode(graph, temp_stack, false);
    synapse_helpers::tensor& syn_tensor = reshapeOp->GetSynOutputs()[0];
    p_context_->syn_inputs_[0] = std::move(syn_tensor);
  }
  ns_TileKernel::ParamsV2 params{};

  auto output = habana_helpers::createPTTensor(
      input,
      RepeatOperator::compute_output_shape(input, repeats),
      input.options(),
      is_output_persistent);

  if (inputs[1].isIntList()) {
    for (int64_t i = 0; i < size; ++i) {
      params.repeat[size - i - 1] = repeats[i];
    }

    // Allocate Shape Tensor
    if (graph.is_dynamic_graph()) {
      auto repeatsShape = habana_helpers::createPTTensor(
          input, repeats, input.options(), false);
      AllocateSynapseShapeTensor(
          graph, repeatsShape, INPUT_DESCRIBING_SHAPE_TENSOR);
    }
  } else {
    TORCH_CHECK(p_context_->syn_inputs_.back().ref().is_input_shape_tensor());
  }

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

at::Tensor repeat_hpu(const at::Tensor& self, at::IntArrayRef repeats) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();
  // Create the operator
  RepeatOperator Op(device_id, scalar_type);
  std::string node_type =
      "tile_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<c10::IValue> stack = {c10::IValue(self), c10::IValue(repeats)};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        RepeatOperator::compute_output_shape(self, repeats),
        self.options(),
        self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  auto output = out.at(0);
  PT_KERNEL_END;
  return output;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::repeat",
            [](const int device_id, c10::ScalarType scalar_type) {
              return std::make_shared<habana::RepeatOperator>(
                  device_id, scalar_type);
            })
        .add(
            "hpu::repeat",
            [](const int device_id, c10::ScalarType scalar_type) {
              return std::make_shared<habana::RepeatOperator>(
                  device_id, scalar_type);
            });
