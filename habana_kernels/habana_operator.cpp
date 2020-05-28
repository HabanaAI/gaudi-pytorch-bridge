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

void habana::HabanaOperator::Compile(synapse_helpers::graph& graph) {
  //
  // compile the graph
  habana_helpers::compile_and_run(
      std::move(graph),
      habana_helpers::names(p_context_->syn_inputs_),
      habana_helpers::names(p_context_->syn_outputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_inputs_),
      habana_helpers::extract_data_ptrs(p_context_->pt_outputs_),
      p_context_->device_id_);
}

synapse_helpers::tensor& habana::HabanaOperator::AllocateSynapseInput(
    synapse_helpers::graph& graph,
    const at::Tensor* input,
    bool is_persistent) {
  TORCH_CHECK(input != nullptr, "Input cannot be null");

  auto syn_tensor_input = habana_helpers::create_tensor(
      *input, graph.get_graph_handle(), is_persistent, c10::nullopt);

  p_context_->syn_inputs_.emplace_back(std::move(syn_tensor_input));

  p_context_->pt_inputs_.emplace_back(input);
  return p_context_->syn_inputs_.back();
}

void habana::HabanaOperator::AllocateSynapseInputs(
    synapse_helpers::graph& graph,
    const std::vector<const at::Tensor*> inputs,
    bool is_persistent) {
  TORCH_CHECK(inputs.size() != 0, "Inputs cannot be null");

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

synapse_helpers::tensor& habana::HabanaOperator::SetSynapseInput(
    synapse_helpers::tensor&& tensor) {
  //
  // The tensor already exists and hence we just add this to the context
  // no need to convert to synapse tensor
  p_context_->syn_inputs_.emplace_back(std::move(tensor));
  return p_context_->syn_inputs_.back();
}

void habana::HabanaOperator::AddNodeToSynapseGraph(
    synapse_helpers::graph& graph,
    void* params,
    size_t params_size) {
  std::vector<synTensor> syn_inputs;
  std::vector<synTensor> syn_outputs;

  for (auto& tensor : p_context_->syn_inputs_) {
    syn_inputs.emplace_back(tensor.get());
  }

  for (auto& tensor : p_context_->syn_outputs_) {
    syn_outputs.emplace_back(tensor.get());
  }

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      params,
      params_size,
      std::move(guid_));
}

habana::HabanaOperator::~HabanaOperator() {}