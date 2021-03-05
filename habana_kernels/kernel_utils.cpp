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

#include <perf_lib_layer_params.h>
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/kernel_recipe_signature.h"
#include "kernel_utils.h"
#include "synapse_helpers/recipe.h"

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
        in_names[i].c_str(),
        reinterpret_cast<uint64_t>(in_buffers[i]),
        DATA_TENSOR,
        {0}});
  for (size_t i = 0; i < out_names.size(); ++i)
    syn_info.emplace_back(synLaunchTensorInfo{
        out_names[i].c_str(),
        reinterpret_cast<uint64_t>(out_buffers[i]),
        DATA_TENSOR,
        {0}});

  return syn_info;
}

std::string habana_helpers::unique_recipe_name_generator(
    std::string recipe_name) {
  static std::unordered_map<std::string, unsigned> map;
  return recipe_name + std::to_string(map[recipe_name]++);
}

static void launchRecipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    std::shared_ptr<synapse_helpers::recipe>& recipe) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto& stream_handle = device.get_compute_stream();
  if (device.IsStreamASyncEnabled()) {
    // wait for input DMA to complete before launching the compute.
    device.add_wait_events_on_stream(in_event_addr, stream_handle);

    auto& recipe_counter = device.get_active_recipe_counter();
    recipe_counter.increase();
    bool status = recipe->launch(input_buffers, output_buffers);
    if (!status) {
      recipe_counter.decrease_and_notify();
      TORCH_CHECK(false, "syn launch failed");
    }
    const auto& recipe_ptr = recipe->getRecipeHandle();
    // Get the reference to the tensor it is operating on to prevent
    // it from being deallocated while the operation is still in flight.
    // so use copy of pt_input in callback
    // regsiter an event on the compute
    device.register_producer_on_stream(
        std::move(out_event_addr),
        stream_handle,
        [pt_inputs, recipe_ptr, &recipe_counter]() {
          recipe_counter.decrease_and_notify();
          return;
        });
  } else {
    recipe->launch(input_buffers, output_buffers);
    TORCH_HABANA_CHECK(
        synStreamSynchronize(stream_handle), "synStreamSynchronize failed");
  }
}

void habana_helpers::compile_and_run(
    synapse_helpers::graph&& graph,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    size_t key) {
  TORCH_CHECK(!graph.is_empty(), "Trying to compile and run an empty graph");

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::shared_ptr<synapse_helpers::recipe> recipe = nullptr;
  if (key > 0 && device.IsCachingEnabled()) {
    recipe = device.get_recipe_handle_cache().get_recipe(key, graph);
  } else {
    recipe = std::make_shared<synapse_helpers::recipe>();
    recipe->create(graph);
  }
  AT_ASSERT(recipe != nullptr);
  if (recipe != nullptr) {
    recipe->create_launch_info();
    recipe->set_inputs_outputs_names(input_names, output_names);
    launchRecipe(
        input_buffers,
        output_buffers,
        in_event_addr,
        out_event_addr,
        pt_inputs,
        device_id,
        recipe);
  }
}

void habana_helpers::execute_recipe(
    const std::vector<void*>& input_buffers,
    const std::vector<void*>& output_buffers,
    std::vector<synapse_helpers::device_ptr> in_event_addr,
    std::vector<synapse_helpers::device_ptr> out_event_addr,
    std::vector<at::Tensor>& pt_inputs,
    const uint32_t device_id,
    size_t key) {
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto recipe = device.get_recipe_handle_cache().get_recipe(key);
  AT_ASSERT(recipe != nullptr);
  if (recipe != nullptr) {
    launchRecipe(
        input_buffers,
        output_buffers,
        in_event_addr,
        out_event_addr,
        pt_inputs,
        device_id,
        recipe);
  }
}

size_t habana_helpers::getRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp,
    bool outOp) {
  RecipeSignature rs(true, stack, {node}, inPlaceOp, outOp);
  return rs.hash();
}

/**
 * @brief CastKernel params structure
 */
ns_CastKernel::Params CastOutOperator::synapse_cast_params_builder() {
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

  auto self = inputs[0].toTensor();
  auto type = inputs[1].toScalarType();
  auto output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      type,
      is_output_persistent);
  inputs.pop_back();
  inputs.push_back(output);
  CastOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

void CastOutOperator::AllocateAndAddSynapseNode(
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

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void ConstantOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  static_cast<void>(is_output_persistent);
  TORCH_CHECK(
      inputs.size() >= 2,
      "Incorrect size of inputs expected for constant operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for constant operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be scalar for constant operator");

  auto output = inputs[0].toTensor();
  auto value = inputs[1].toScalar();

  TORCH_CHECK(
      (output.scalar_type() == c10::ScalarType::BFloat16) ||
          (output.scalar_type() == c10::ScalarType::Int) ||
          (output.scalar_type() == c10::ScalarType::Float),
      "Unsupported dtype provided for ConstantOut kernel Input.scalar_type() = ",
      output.scalar_type());

  ns_ConstantKernel::Params params;
  if (output.scalar_type() == c10::ScalarType::Int) {
    params.constant.i = value.to<int32_t>();
  } else {
    params.constant.f = value.to<float>();
  }

  p_context_->params_.emplace<ns_ConstantKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (output.dim() == 0) {
    output.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  // Note that Constant TPC kernel does not need any tensor inputs
  // therefore we can move the input tensor(s) to corresponding
  // output tensors without any problems.
  HABANA_ASSERT(p_context_->syn_inputs_.size() == 1);
  synapse_helpers::tensor_or_ref& input_tensor = p_context_->syn_inputs_.back();
  p_context_->syn_outputs_.emplace_back(std::move(input_tensor));
  p_context_->pt_outputs_.emplace_back(output);
  // Adding a clear for inputs as constant kernel expects no inputs
  // AS we get inputs from PT kernel, graph mode creates a syn tensor anyway
  // It was observed if we let that syn tensor remain, the kernel gives wrong
  // outputs
  p_context_->syn_inputs_.clear();
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void ConstantOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() >= 2,
      "Incorrect size of inputs expected for constant operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for constant operator");
  TORCH_CHECK(
      inputs[1].isScalar(),
      "Input arg2 expected to be scalar for constant operator");

  auto input = inputs[0].toTensor();
  auto value = inputs[1].toScalar();

  TORCH_CHECK(
      (input.scalar_type() == c10::ScalarType::BFloat16) ||
          (input.scalar_type() == c10::ScalarType::Int) ||
          (input.scalar_type() == c10::ScalarType::Char) ||
          (input.scalar_type() == c10::ScalarType::Float),
      "Unsupported dtype provided for Constant kernel Input.scalar_type() = ",
      input.scalar_type());

  ns_ConstantKernel::Params params;
  if (input.scalar_type() == c10::ScalarType::Int) {
    params.constant.i = value.to<int32_t>();
  } else {
    params.constant.f = value.to<float>();
  }

  p_context_->params_.emplace<ns_ConstantKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  if (input.dim() == 0) {
    input.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  }

  auto output = habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  // Adding a clear for inputs as constant kernel expects no inputs
  // AS we get inputs from PT kernel, graph mode creates a syn tensor anyway
  // It was observed if we let that syn tensor remain, the kernel gives wrong
  // outputs
  p_context_->syn_inputs_.clear();
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

static auto& KernelRegistry = habana::KernelRegistry().add(
    "aten::ones_like",
    [](const int device_id, c10::ScalarType node_type) {
      return std::make_shared<OnesLikeOperator>(device_id, node_type);
    });
