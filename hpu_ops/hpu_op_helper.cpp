/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "hpu_op_helper.h"
#include "generated/hpu_op.h"

namespace habana {

std::vector<at::Tensor> GetMetaTensorList(
    const std::vector<at::Tensor>& tensors) {
  std::vector<at::Tensor> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    metatensors.emplace_back(at::empty_meta(
        tensor.sizes(), tensor.options(), tensor.suggest_memory_format()));
  }
  return metatensors;
}

std::vector<c10::optional<at::Tensor>> GetMetaOptTensorList(
    const std::vector<c10::optional<at::Tensor>>& tensors) {
  std::vector<c10::optional<at::Tensor>> metatensors;
  metatensors.reserve(tensors.size());
  for (const auto& tensor : tensors) {
    if (tensor.has_value()) {
      const auto& tv = tensor.value();
      metatensors.emplace_back(
          at::empty_meta(tv.sizes(), tv.options(), tv.suggest_memory_format()));
    } else {
      metatensors.emplace_back(tensor);
    }
  }
  return metatensors;
}

void HabanaOperatorHelper::HandleScalarToTensor(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (m_scalar_id < 0) {
    return;
  }

  at::Scalar val = stack.at(m_scalar_id).toScalar();
  m_scalar_inputs.emplace(m_scalar_id, val);

  size_t size = 0;
  PARAMS_STUB(ns_ConstantKernel::Params);
  if (m_scalar_type == c10::ScalarType::Int) {
    get<int>(params->constant) = val.to<int>();
  } else {
    get<float>(params->constant) = val.to<float>();
  }

  auto const_out = BuildOp(
      graph,
      "constant_" + habana_helpers::name_suffix_from_type(m_scalar_type),
      {},
      {{1, m_scalar_type}},
      params.get(),
      size);

  // Set output from constant as input to this node at index m_scalar_id
  p_context_->syn_inputs_.emplace(
      p_context_->syn_inputs_.cbegin() + m_scalar_id, std::move(const_out[0]));
}

void HabanaOperatorHelper::HandleFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  if (m_out_id < 0) {
    return;
  }

  for (const auto& is_output_persistent : is_output_persistent_list) {
    const auto& output = habana_helpers::createPTTensor(
        stack.at(m_out_id).toTensor(), is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
}

void HabanaOperatorHelper::HandleOutFn(const at::Stack& stack) {
  if (!m_is_outfn) {
    return;
  }

  p_context_->pt_outputs_.emplace_back(stack.back().toTensor());
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_.back()));
  p_context_->syn_inputs_.pop_back();
}

void HabanaOperatorHelper::HandleInplaceFn(const at::Stack& stack) {
  if (m_inplace_id < 0) {
    return;
  }

  // Index can vary in syn_inputs_ and in stack
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[m_inplace_id]));
  p_context_->pt_outputs_.emplace_back(stack[m_inplace_id].toTensor());
}

void HabanaOperatorHelper::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>&) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  AddNodeToSynapseGraph(graph, params.get(), size);
}

void HabanaOperatorHelper::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    std::vector<bool> is_output_persistent_list) {
  CustomHandler(graph, stack);
  HandleFn(graph, stack, is_output_persistent_list);
  HandleInplaceFn(stack);
  HandleOutFn(stack);
  HandleScalarToTensor(graph, stack);

  AddNode(graph, stack, is_output_persistent_list);
}

std::vector<synapse_helpers::tensor> HabanaOperatorHelper::BuildOp(
    synapse_helpers::graph& graph,
    std::string guid,
    std::vector<synTensor> node_inputs,
    const std::vector<_node_output_attr>& node_output_attrs,
    void* params,
    size_t param_size) {
  std::vector<synapse_helpers::tensor> outputs;
  std::vector<synTensor> node_outputs;

  for (const auto& attr : node_output_attrs) {
    if (attr.synout_index < 0) {
      const auto& t = at::detail::make_tensor<c10::TensorImpl>(
          c10::DispatchKeySet{
              at::DispatchKey::HPU, at::DispatchKey::AutogradHABANA},
          c10::scalarTypeToTypeMeta(attr.dtype),
          c10::Device(c10::kHABANA, 0));
      t.unsafeGetTensorImpl()->set_sizes_contiguous(attr.sizes);
      outputs.emplace_back(habana_helpers::create_tensor(
          t, graph.get_graph_handle(), attr.persistent, attr.dtype));
      node_outputs.emplace_back(outputs.back().get());
    } else {
      outputs.emplace_back(
          std::move(p_context_->syn_outputs_.at(attr.synout_index).ref()));
      node_outputs.emplace_back(outputs.back().get());
    }
  }

  auto result = graph.add_node(
      std::move(node_inputs),
      std::move(node_outputs),
      params,
      param_size,
      guid);
  HABANA_ASSERT(
      ok(result),
      "Adding ",
      guid,
      " to graph failed with ",
      get_error(result).error);

  return outputs;
}
} // namespace habana
