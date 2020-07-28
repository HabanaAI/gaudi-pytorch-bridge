/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_operator.h"
#include "habana_kernels/kernel_utils.h"

const at::IntArrayRef& habana::HabanaOperator::getPermuteOrder(
    const LayoutFormat target_layout,
    bool to_device) {
  static const std::unordered_map<const LayoutFormat, const at::IntArrayRef>
      toDevicePermuteOrder = {
          // Host -> Device
          {LayoutFormat::NHWC, {0, 2, 3, 1}}, // NCHW -> NHWC
          {LayoutFormat::NCHW, {0, 1, 2, 3}}, // NCHW -> NCHW (No Change)
          {LayoutFormat::HWCK, {2, 3, 1, 0}}, // KCHW -> HWCK
          {LayoutFormat::ANY, {0, 1, 2, 3}} // XXXX -> XXXX (No Change)
      };

  static const std::unordered_map<const LayoutFormat, const at::IntArrayRef>
      toHostPermuteOrder = {
          // Device -> Host
          {LayoutFormat::NCHW, {0, 1, 2, 3}}, // NCHW   -> NCHW (No Change)
          {LayoutFormat::NHWC, {0, 3, 1, 2}}, // NHWC   -> NCHW
          {LayoutFormat::HWCK, {3, 2, 0, 1}}, // HWCK   -> KCHW
          {LayoutFormat::ANY, {0, 1, 2, 3}} // XXXX   -> XXXX (No Change)
      };

  const auto& permuteOrder =
      (to_device ? toDevicePermuteOrder : toHostPermuteOrder);

  TORCH_CHECK(
      permuteOrder.find(target_layout) != permuteOrder.end(),
      "Unknown layout in getPermuteOrder");
  return permuteOrder.find(target_layout)->second;
}

void habana::HabanaOperator::Compile(synapse_helpers::graph& graph) {
  //
  // compile the graph
  habana_helpers::compile_and_run(
      std::move(graph),
      habana_helpers::names(p_context_->syn_inputs_),
      habana_helpers::names(p_context_->syn_outputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_outputs_),
      p_context_->pt_inputs_,
      p_context_->device_id_,
      p_context_->recipe_key_);
}

void habana::HabanaOperator::Execute(size_t key) {
  //
  // Execute the graph
  habana_helpers::execute_recipe(
      habana_helpers::extract_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_outputs_),
      p_context_->pt_inputs_,
      p_context_->device_id_,
      p_context_->recipe_key_);
}

void habana::HabanaOperator::SetPTInputs(
    const std::vector<at::Tensor>& inputs) {
  for (auto& input : inputs) {
    p_context_->pt_inputs_.emplace_back(input);
  }
}

void habana::HabanaOperator::SetPTOutput(const at::Tensor& output) {
  p_context_->pt_outputs_.emplace_back(output);
}

void habana::HabanaOperator::SetPTOutputs(
    const std::vector<at::Tensor>& outputs) {
  TORCH_CHECK(outputs.size() != 0, "Outputs cannot be null");

  for (auto& output : outputs) {
    p_context_->pt_outputs_.emplace_back(output);
  }
}

size_t habana::HabanaOperator::GetRecipeKey(
    std::string node,
    std::vector<c10::IValue> stack,
    bool inPlaceOp,
    bool outOp) {
  size_t key = habana_helpers::getRecipeKey(node, stack, inPlaceOp, outOp);
  p_context_->recipe_key_ = key;
  return key;
}

synapse_helpers::tensor& habana::HabanaOperator::AllocateSynapseInput(
    synapse_helpers::graph& graph,
    const at::Tensor& input,
    bool is_persistent) {
  // TORCH_CHECK(input != nullptr, "Input cannot be null");

  auto syn_tensor_input = habana_helpers::create_tensor(
      input, graph.get_graph_handle(), is_persistent, c10::nullopt);

  p_context_->syn_inputs_.emplace_back(std::move(syn_tensor_input));

  p_context_->pt_inputs_.emplace_back(input);
  return p_context_->syn_inputs_.back();
}

void habana::HabanaOperator::AllocateSynapseInputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& inputs,
    bool is_persistent) {
  // TORCH_CHECK(!inputs.empty(), "Inputs cannot be null");

  for (auto& input : inputs) {
    AllocateSynapseInput(graph, input, is_persistent);
  }
}

void habana::HabanaOperator::AllocateSynapseOutput(
    synapse_helpers::graph& graph,
    const at::Tensor& output,
    bool is_persistent) {
  p_context_->syn_outputs_.emplace_back(habana_helpers::create_tensor(
      output, graph.get_graph_handle(), is_persistent, c10::nullopt));

  p_context_->pt_outputs_.emplace_back(output);
}

std::vector<std::pair<std::string, at::Tensor>> habana::HabanaOperator::getAppendedTensorInfos()
{
  return appended_tensor_infos;
}

void habana::HabanaOperator::AllocateSynapseInplaceOutput(
    synapse_helpers::graph& graph) {
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0]));

  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
}

void habana::HabanaOperator::AllocateSynapseOutputs(
    synapse_helpers::graph& graph,
    const std::vector<at::Tensor>& outputs,
    bool is_persistent) {
  TORCH_CHECK(outputs.size() != 0, "Outputs cannot be null");

  for (auto& output : outputs) {
    AllocateSynapseOutput(graph, output, is_persistent);
  }
}

synapse_helpers::tensor_or_ref& habana::HabanaOperator::SetSynapseInput(
    synapse_helpers::tensor_or_ref&& tensor) {
  //
  // The tensor already exists and hence we just add this to the context
  // no need to convert to synapse tensor
  p_context_->syn_inputs_.emplace_back(std::move(tensor));
  return p_context_->syn_inputs_.back();
}

synapse_helpers::tensor_or_ref& habana::HabanaOperator::SetSynapseOutput(
    synapse_helpers::tensor_or_ref&& tensor) {
  //
  // The tensor already exists and hence we just add this to the context
  // no need to convert to synapse tensor
  p_context_->syn_outputs_.emplace_back(std::move(tensor));
  return p_context_->syn_outputs_.back();
}

void habana::HabanaOperator::AddNodeToSynapseGraph(
    synapse_helpers::graph& graph,
    void* params,
    size_t params_size) {
  std::vector<synTensor> syn_inputs;
  std::vector<synTensor> syn_outputs;

  for (size_t i = 0; i < p_context_->syn_inputs_.size(); i++) {
    synapse_helpers::tensor& tensor = p_context_->syn_inputs_[i];
    if (kernel_meta_data_.valid_input_idx.empty() ||
        kernel_meta_data_.valid_input_idx.count(i)) {
      syn_inputs.emplace_back(tensor.get());
    }
  }

  for (synapse_helpers::tensor& tensor : p_context_->syn_outputs_) {
    syn_outputs.emplace_back(tensor.get());
  }

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      params,
      params_size,
      std::move(guid_));
}

habana::RegisterKernel& habana::KernelRegistry() {
  static habana::RegisterKernel* Registry = new habana::RegisterKernel();
  return *Registry;
}

habana::HabanaOperator::~HabanaOperator() = default;
