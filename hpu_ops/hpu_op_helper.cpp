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

bool SupportedDtypes::count(c10::ScalarType type) const {
  return m_dtypes.count(type);
}

bool SupportedDtypes::count(const at::Tensor& tensor) const {
  return count(tensor.scalar_type());
}

bool SupportedDtypes::count(const c10::optional<at::Tensor>& tensor) const {
  return tensor.has_value() and count(tensor.value());
}

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

static at::ScalarType GetScalarType(const at::Stack& stack, int index) {
  const auto& ival = stack.at(index);
  auto type =
      ival.isTensor() ? ival.toTensor().scalar_type() : ival.toScalar().type();
  if (type == at::ScalarType::Long) {
    return at::ScalarType::Int;
  } else if (type == at::ScalarType::Double) {
    return at::ScalarType::Float;
  } else if (type == at::ScalarType::Bool) {
    return at::ScalarType::Char;
  }
  return type;
}

OpBackend::OpBackend(
    int device_id,
    const std::string& guid,
    c10::ScalarType scalar_type,
    std::vector<int> res_ids,
    std::vector<int> inplace_ids,
    std::vector<int> scalar_ids,
    bool is_outfn)
    : HabanaOperator(guid + habana_helpers::name_suffix_from_type(scalar_type)),
      m_res_ids{std::move(res_ids)},
      m_inplace_ids{std::move(inplace_ids)},
      m_scalar_ids{std::move(scalar_ids)},
      m_is_outfn{is_outfn},
      m_scalar_type{scalar_type} {
  CreateSynContext(device_id);
  kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
}

void OpBackend::HandleScalarToTensor(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (m_scalar_ids.empty()) {
    return;
  }

  for (int m_scalar_id : m_scalar_ids) {
    const at::Scalar& val = stack.at(m_scalar_id).toScalar();
    m_scalar_inputs.emplace(m_scalar_id, val);

    auto constant = ConstantHelper(graph, val);

    // Set output from constant as input to this node at index m_scalar_id
    p_context_->syn_inputs_.emplace(
        p_context_->syn_inputs_.cbegin() + m_scalar_id, std::move(constant));
  }
}

void OpBackend::HandleFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  if (m_res_ids.empty()) {
    return;
  }

  const auto& outshapes = ComputeOutputShapes(stack, true);

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      outshapes.empty() || outshapes.size() == is_output_persistent_list.size(),
      "Num outputs and num outshapes does not match ",
      is_output_persistent_list.size(),
      " != ",
      outshapes.size());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      m_res_ids.size() == is_output_persistent_list.size(),
      "Num outputs defined (",
      m_res_ids.size(),
      ") as out_ids is not matching with actual num outputs (",
      is_output_persistent_list.size());
  for (unsigned i = 0; i < is_output_persistent_list.size(); ++i) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        stack.at(m_res_ids.at(i)).isTensor(),
        "Index in out_ids[",
        i,
        "] is incorrect, got ",
        stack.at(m_res_ids.at(i)).tagKind(),
        " instead of Tensor.");
    // Use sizes of tensor at m_res_id if ComputeOutputShapes() is not
    // implemented
    const auto& t = stack.at(m_res_ids.at(i)).toTensor();
    const auto& dtype = m_promote_type
        ? at::promote_types(GetScalarType(stack, 0), GetScalarType(stack, 1))
        : t.scalar_type();
    const auto& outshape = outshapes.empty() ? t.sizes() : outshapes[i];
    bool is_output_persistent = is_output_persistent_list[i];

    const auto& output = habana_helpers::createPTTensor(
        t, outshape, t.options().dtype(dtype), is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
  }
}

void OpBackend::HandleOutFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (!m_is_outfn) {
    return;
  }

  unsigned stack_size = stack.size();
  unsigned syn_inputs_size = p_context_->syn_inputs_.size();

  for (int i = m_num_out_tensors; i > 0; --i) {
    p_context_->pt_outputs_.emplace_back(stack.at(stack_size - i).toTensor());
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_.at(syn_inputs_size - i), graph));
  }

  // Remove the out tensors from syn inputs
  p_context_->syn_inputs_.erase(
      p_context_->syn_inputs_.end() - m_num_out_tensors,
      p_context_->syn_inputs_.end());
}

void OpBackend::HandleInplaceFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (m_inplace_ids.empty()) {
    return;
  }

  for (int inplace_id : m_inplace_ids) {
    // Index can vary in syn_inputs_ and in stack
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[inplace_id], graph));
    p_context_->pt_outputs_.emplace_back(stack[inplace_id].toTensor());
  }
}

void OpBackend::HandleTypePromotion(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (!m_promote_type) {
    return;
  }

  const std::array<at::ScalarType, 2> input_types{
      GetScalarType(stack, 0), GetScalarType(stack, 1)};
  const at::ScalarType& result_type =
      at::promote_types(input_types[0], input_types[1]);

  int cast_index = -1;
  if (input_types[0] != result_type and input_types[1] == result_type) {
    cast_index = 0;
  } else if (input_types[0] == result_type and input_types[1] != result_type) {
    cast_index = 1;
  } else {
    // No cast needed
    return;
  }

  std::vector<synTensor> syn_inputs{syn_in(0), syn_in(1)};
  // Insert cast on the input with lower dtype
  auto cast = CastHelper(
      graph,
      syn_inputs.at(cast_index),
      stack.at(cast_index).isTensor() ? stack_tensor(stack, cast_index).sizes()
                                      : 1,
      input_types[cast_index],
      result_type);

  // Replace the input with the casted input
  p_context_->syn_inputs_.at(cast_index) = std::move(cast);

  // Update the guid to reflect the promoted type
  SetGuid(
      guid_.substr(0, guid_.find_last_of('_') + 1) +
      habana_helpers::name_suffix_from_type(result_type));

  // Update m_scalar_type
  m_scalar_type = result_type;
}

std::vector<synapse_helpers::tensor> OpBackend::BuildOp(
    synapse_helpers::graph& graph,
    const std::string& guid,
    std::vector<synTensor> node_inputs,
    const std::vector<NodeAttr::NodeOutputAttr>& node_output_attr,
    void* params,
    size_t param_size) {
  return OpBackend::BuildNode(
      this,
      graph,
      {guid, std::move(node_inputs), node_output_attr, params, param_size});
}

synapse_helpers::tensor OpBackend::CastHelper(
    synapse_helpers::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    const at::ScalarType& from,
    const at::ScalarType& to,
    bool persistent,
    bool final_node) {
  return OpBackend::BuildCast(
      this,
      graph,
      syn_in,
      sizes,
      from,
      to,
      CAST_ROUND_HALF_NE,
      persistent,
      final_node);
}

synapse_helpers::tensor OpBackend::ConstantHelper(
    synapse_helpers::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    const at::IntArrayRef constant_outshape,
    bool persistent,
    bool final_node) {
  return OpBackend::BuildConstant(
      this, graph, val, force_type, constant_outshape, persistent, final_node);
}

void OpBackend::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>&) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  AddNodeToSynapseGraph(graph, params.get(), size);
}

void OpBackend::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    std::vector<bool> is_output_persistent_list) {
  CustomHandler(graph, stack);

  HandleFn(graph, stack, is_output_persistent_list);
  HandleInplaceFn(graph, stack);
  HandleOutFn(graph, stack);

  HandleScalarToTensor(graph, stack);
  HandleTypePromotion(graph, stack);

  AddNode(graph, stack, is_output_persistent_list);
}

std::vector<synapse_helpers::tensor> OpBackend::BuildNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    NodeAttr node_attr) {
  auto ctx = op->p_context_;
  std::vector<synapse_helpers::tensor> outputs;
  std::vector<synTensor> node_outputs;
  int available_output_id = 0;
  int persistent_output_id = 0;
  int final_output_id = 0;

  for (const auto& attr : node_attr.output_attrs) {
    if (attr.final_node and op->IsOutputAvailable()) {
      // HandleOutFn/HandleInplaceFn placed the output in syn_outputs_
      outputs.emplace_back(
          std::move(ctx->syn_outputs_.at(available_output_id++).ref()));
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
        const auto& impl =
            ctx->pt_outputs_.at(persistent_output_id++).unsafeGetTensorImpl();
        impl->set_sizes_contiguous(attr.sizes);
        impl->set_storage_and_dtype(
            impl->storage(), c10::scalarTypeToTypeMeta(attr.dtype));
      } else if (attr.final_node) {
        ctx->pt_outputs_.at(final_output_id++) = t;
      }
    }
    node_outputs.emplace_back(outputs.back().get());
  }

  auto result = graph.add_node(
      std::move(node_attr.inputs),
      std::move(node_outputs),
      node_attr.params,
      node_attr.param_size,
      node_attr.guid);
  HABANA_ASSERT(
      ok(result),
      "Adding ",
      node_attr.guid,
      " to graph failed with ",
      get_error(result).error);

  return outputs;
}

synapse_helpers::tensor OpBackend::BuildCast(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    const at::IntArrayRef sizes,
    const at::ScalarType& from,
    const at::ScalarType& to,
    CastF32RoundMode_t round_mode,
    bool persistent,
    bool final_node) {
  const auto& guid = "cast_" + habana_helpers::name_suffix_from_type(from) +
      "_to_" + habana_helpers::name_suffix_from_type(to);
  ns_CastKernel::Params params{round_mode};
  NodeAttr castnode{
      guid,
      {syn_in},
      {{sizes, to, persistent, final_node}},
      &params,
      sizeof(params)};
  auto cast = BuildNode(op, graph, std::move(castnode));

  return std::move(cast.at(0));
}

synapse_helpers::tensor OpBackend::BuildConstant(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    const at::IntArrayRef constant_outshape,
    bool persistent,
    bool final_node) {
  const at::ScalarType& valtype =
      force_type.has_value() ? force_type.value() : val.type();
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      at::canCast(val.type(), valtype),
      __func__,
      " cannot cast ",
      val.type(),
      " (",
      val.isFloatingPoint() ? val.toFloat() : val.toInt(),
      ") to ",
      valtype);

  ns_ConstantKernel::Params params{};
  if (valtype == c10::ScalarType::Int or valtype == c10::ScalarType::Long) {
    get<int>(params.constant) = val.to<int>();
  } else {
    get<float>(params.constant) = val.to<float>();
  }

  auto constant = BuildNode(
      op,
      graph,
      {"constant_" + habana_helpers::name_suffix_from_type(valtype),
       {},
       {{constant_outshape, valtype, persistent, final_node}},
       &params,
       sizeof(params)});
  return std::move(constant.at(0));
}
} // namespace habana
