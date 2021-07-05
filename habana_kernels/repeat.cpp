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

void RepeatOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for repeat operator");
  TORCH_CHECK(
      inputs[1].isIntList(),
      "Input arg2 expected to be intlist for repeat operator");
  auto input = inputs[0].toTensor();
  auto repeats = inputs[1].toIntVector();
  ns_TileKernel::ParamsV2 params{};

  int64_t size = repeats.size();
  for (int64_t i = 0; i < size; ++i) {
    params.repeat[size - i - 1] = repeats[i];
  }

  auto output = habana_helpers::createPTTensor(
      input,
      RepeatOperator::compute_output_shape(input, repeats),
      input.options(),
      is_output_persistent);
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

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::repeat",
    [](const int device_id, c10::ScalarType scalar_type) {
      return std::make_shared<habana::RepeatOperator>(device_id, scalar_type);
    });
