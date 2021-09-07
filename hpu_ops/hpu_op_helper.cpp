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
    metatensors.emplace_back(at::empty(
        tensor.sizes(),
        tensor.options().device(at::kMeta),
        tensor.suggest_memory_format()));
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
      metatensors.emplace_back(at::empty(
          tv.sizes(),
          tv.options().device(at::kMeta),
          tv.suggest_memory_format()));
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
  // TODO Handle multiple outputs
  if (m_out_id < 0) {
    return;
  }

  const auto& outshapes = ComputeOutputShapes(stack);
  const auto& t = stack.at(m_out_id).toTensor();

  HABANA_ASSERT(
      outshapes.empty() || outshapes.size() == is_output_persistent_list.size(),
      "Num outputs and num outshapes does not match ",
      is_output_persistent_list.size(),
      " != ",
      outshapes.size());

  for (unsigned i = 0; i < is_output_persistent_list.size(); ++i) {
    // Use sizes of tensor at m_out_id if ComputeOutputShapes() is not
    // implemented
    const auto& outshape = outshapes.empty() ? t.sizes() : outshapes[i];
    bool is_output_persistent = is_output_persistent_list[i];
    const auto& output = habana_helpers::createPTTensor(
        t, outshape, t.options(), is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
}

void HabanaOperatorHelper::HandleOutFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (!m_is_outfn) {
    return;
  }

  p_context_->pt_outputs_.emplace_back(stack.back().toTensor());
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_.back(), graph));
  p_context_->syn_inputs_.pop_back();
}

void HabanaOperatorHelper::HandleInplaceFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (m_inplace_id < 0) {
    return;
  }

  // Index can vary in syn_inputs_ and in stack
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[m_inplace_id], graph));
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
  HandleInplaceFn(graph, stack);
  HandleOutFn(graph, stack);
  HandleScalarToTensor(graph, stack);

  AddNode(graph, stack, is_output_persistent_list);
}

std::vector<synapse_helpers::tensor> HabanaOperatorHelper::BuildOp(
    synapse_helpers::graph& graph,
    const std::string& guid,
    std::vector<synTensor> node_inputs,
    const std::vector<_node_output_attr>& node_output_attrs,
    void* params,
    size_t param_size) {
  std::vector<synapse_helpers::tensor> outputs;
  std::vector<synTensor> node_outputs;

  for (const auto& attr : node_output_attrs) {
    if (attr.final_node and IsOutFn()) {
      // HandleOutFn() placed the output in p_context_->syn_outputs_
      outputs.emplace_back(std::move(p_context_->syn_outputs_.at(0).ref()));
    } else {
      const auto& t = at::detail::make_tensor<c10::TensorImpl>(
          c10::DispatchKeySet{
              at::DispatchKey::HPU, at::DispatchKey::AutogradHPU},
          c10::scalarTypeToTypeMeta(attr.dtype),
          c10::Device(c10::kHPU, 0));
      t.unsafeGetTensorImpl()->set_sizes_contiguous(attr.sizes);
      outputs.emplace_back(
          habana_helpers::create_tensor(t, graph, attr.persistent, attr.dtype));
      if (attr.persistent) {
        // TODO: Handle when a node produces multiple outputs
        HABANA_ASSERT(
            m_out_id >= 0, "Out id cannot be negative for persistent output");
        const auto& impl =
            p_context_->pt_outputs_.at(m_out_id).unsafeGetTensorImpl();
        impl->set_sizes_contiguous(attr.sizes);
        impl->set_storage_and_dtype(
            impl->storage(), c10::scalarTypeToTypeMeta(attr.dtype));
      }
    }
    node_outputs.emplace_back(outputs.back().get());
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
