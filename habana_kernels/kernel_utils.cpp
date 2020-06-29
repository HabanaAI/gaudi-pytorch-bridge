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
#include "habana_kernels/kernel_recipe_signature.h"
#include "kernel_utils.h"
#include "synapse_helpers/recipe.h"
#include <perf_lib_layer_params.h>

using namespace torch;

std::vector<synLaunchTensorInfo> habana_helpers::
    generate_syn_launch_tensor_info(
        const std::vector<std::string>& in_names,
        const std::vector<void*>& in_buffers,
        const std::vector<std::string>& out_names,
        const std::vector<void*>& out_buffers) {
  TORCH_CHECK(in_names.size() == in_buffers.size());
  TORCH_CHECK(out_names.size() == out_buffers.size());

  std::vector<synLaunchTensorInfo> syn_info;
  syn_info.reserve(in_names.size() + out_names.size());

  for (size_t i = 0; i < in_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        in_names[i].c_str(), reinterpret_cast<uint64_t>(in_buffers[i])});
  for (size_t i = 0; i < out_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        out_names[i].c_str(), reinterpret_cast<uint64_t>(out_buffers[i])});

  return syn_info;
}

std::string habana_helpers::unique_recipe_name_generator(
    std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
}

void habana_helpers::compile_and_run(
    synapse_helpers::graph&& graph,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id,
    size_t key) {
  TORCH_CHECK(!graph.is_empty(), "Trying to compile and run an empty graph");

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::shared_ptr<synapse_helpers::recipe> recipe = nullptr;
  if (key > 0 && synapse_helpers::IsCachingEnabled()) {
    recipe = device.get_recipe_handle_cache().get_recipe(key, graph);
  } else {
    recipe = std::make_shared<synapse_helpers::recipe>();
    recipe->create(graph);
  }
  AT_ASSERT(recipe != nullptr);
  if (recipe != nullptr) {
    synStreamHandle stream_handle = device.get_compute_stream();
    recipe->create_launch_info();
    recipe->set_inputs_outputs_names(input_names, output_names);
    recipe->launch(input_buffers, output_buffers);
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

void habana_helpers::execute_recipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    const uint32_t device_id,
    size_t key) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  synStreamHandle stream_handle = device.get_compute_stream();
  auto g_recipe = device.get_recipe_handle_cache().get_recipe(key);
  AT_ASSERT(g_recipe != nullptr);
  if (g_recipe != nullptr) {
    g_recipe->launch(input_buffers, output_buffers);
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

size_t habana_helpers::getRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp) {
  RecipeSignature rs(true, stack, {node}, inPlaceOp);
  return rs.hash();
}

namespace habana {

/**
 * @brief CastKernel params structure
 */
ns_CastKernel::Params CastOperator::synapse_cast_params_builder() {
  ns_CastKernel::Params cast_params{};
  cast_params.round_mode = CAST_ROUND_HALF_NE;

  return cast_params;
}

void CastOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for cast operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for cast operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for cast operator");

  auto self = inputs[0].toTensor();
  auto output = inputs[1].toTensor();

  ns_CastKernel::Params params = synapse_cast_params_builder();
  p_context_->params_.emplace<ns_CastKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  std::vector<at::Tensor> outputs{output};
  AllocateSynapseOutputs(graph, outputs, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

} // namespace habana