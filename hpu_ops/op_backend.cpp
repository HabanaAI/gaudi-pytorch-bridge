/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "op_backend.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_kernels/kernel_utils.h"
#include "hpu_op_helper.h"

namespace habana {
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

static at::Tensor GetProxyTensor(at::ScalarType dtype, at::IntArrayRef sizes) {
  const auto& t = at::detail::make_tensor<c10::TensorImpl>(
      c10::DispatchKeySet{at::DispatchKey::HPU, at::DispatchKey::AutogradHPU},
      c10::scalarTypeToTypeMeta(dtype),
      c10::Device(c10::kHPU, 0));
  t.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);

  return t;
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

synTensor OpBackend::syn_in(int index) {
  if (isMetaMode()) {
    return nullptr;
  }
  return p_context_->syn_inputs_.at(index).ref().get();
}

synapse_helpers::tensor& OpBackend::syn_out(int index) {
  if (isMetaMode()) {
    static auto ph = synapse_helpers::tensor::create_placeholder(0, {}, {});
    return ph;
  }
  return p_context_->syn_outputs_.at(index);
}

c10::ScalarType OpBackend::ComputePromotedScalarType(
    const at::Stack& stack,
    bool update) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_promote_type or m_promote_int_to_float);
  habana_helpers::DTypeHelper dtype_helper;
  dtype_helper.add_inputs({&stack.at(0), &stack.at(1)})
      .set_promote_to_common_type(m_promote_type)
      .set_promote_int_to_float(m_promote_int_to_float)
      .build();
  c10::ScalarType result_type = dtype_helper.get_result_dtype();

  if (update) {
    m_scalar_type = result_type;
  }

  return result_type;
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
    const at::Stack& stack) {
  if (m_res_ids.empty()) {
    return;
  }

  std::vector<at::Tensor> tensors;
  const auto& outshapes = ComputeOutputShapes(stack, true);

  for (int res_id : m_res_ids) {
    at::IValue ival = stack.at(res_id);
    if (ival.isTensor()) {
      tensors.emplace_back(ival.toTensor());
    } else if (ival.isTensorList()) {
      const auto& list = ival.toTensorList();
      tensors.insert(tensors.end(), list.begin(), list.end());
    } else {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          false,
          "Result type can be only tensor or a list of tensors but got ",
          ival.tagKind());
    }
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      outshapes.empty() || outshapes.size() == m_output_metadata.size(),
      "Num outputs and num outshapes does not match ",
      m_output_metadata.size(),
      " != ",
      outshapes.size());

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      tensors.size() == m_output_metadata.size(),
      "Num outputs defined (",
      m_res_ids.size(),
      ") as out_ids is not matching with actual num outputs (",
      m_output_metadata.size());

  int i = 0;
  for (const at::Tensor& t : tensors) {
    const auto& dtype = m_promote_type or m_promote_int_to_float
        ? ComputePromotedScalarType(stack, true)
        : t.scalar_type();
    const auto& outshape = outshapes.empty() ? t.sizes() : outshapes[i];

    const auto& output = habana_helpers::createPTTensor(
        t,
        outshape,
        t.options().dtype(dtype),
        m_output_metadata.at(i).persistent);
    AllocateSynapseOutput(graph, output, m_output_metadata.at(i++));
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
            p_context_->syn_inputs_.at(syn_inputs_size - i),
            graph,
            m_output_metadata.at(m_num_out_tensors - i).external));
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
    const auto& ival = stack[inplace_id];
    const auto& tensors = ival.isTensor()
        ? static_cast<at::List<at::Tensor>>(ival.toTensor())
        : ival.toTensorList();
    for (auto i = 0u; i < tensors.size(); ++i) {
      p_context_->syn_outputs_.emplace_back(
          habana_helpers::duplicate_tensor_in_memory_section(
              p_context_->syn_inputs_[i],
              graph,
              m_output_metadata.at(inplace_id).external));
      p_context_->pt_outputs_.emplace_back(tensors[i]);
    }
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
  const at::ScalarType& result_type = ComputePromotedScalarType(stack, true);

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
}

void OpBackend::HandleIntToFloatPromotion(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  if (!m_promote_int_to_float) {
    return;
  }

  const std::array<at::ScalarType, 2> input_types{
      GetScalarType(stack, 0), GetScalarType(stack, 1)};
  const at::ScalarType& result_type = ComputePromotedScalarType(stack, true);

  for (auto i = 0u; i < input_types.size(); ++i) {
    if (input_types[i] == result_type) {
      continue;
    }

    auto cast = CastHelper(
        graph,
        syn_in(i),
        stack.at(i).isTensor() ? stack_tensor(stack, i).sizes() : 1,
        input_types[i],
        result_type);

    // Replace the input with the casted input
    p_context_->syn_inputs_.at(i) = std::move(cast);
  }

  // Update the guid to reflect the promoted type
  update_guid_dtype(guid_, habana_helpers::name_suffix_from_type(result_type));
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
    c10::optional<int> final_result_index) {
  return OpBackend::BuildCast(
      this, graph, syn_in, sizes, from, to, final_result_index);
}

synapse_helpers::tensor OpBackend::ConstantHelper(
    synapse_helpers::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    const at::IntArrayRef constant_outshape,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildConstant(
      this, graph, val, force_type, constant_outshape, final_result_index);
}

synapse_helpers::tensor OpBackend::ReshapeHelper(
    synapse_helpers::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildReshape(
      this, graph, syn_in, sizes, dtype, final_result_index);
}

void OpBackend::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  if (isMetaMode()) {
    const auto& t = stack[0].toTensor();
    const auto& sizes = m_compute_output_shapes
        ? m_compute_output_shapes(stack, true)[0]
        : t.sizes().vec();
    m_meta.AddOutputTensor(TensorMetaData(
        sizes, t.strides().vec(), t.scalar_type(), t.suggest_memory_format()));
    return;
  }
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  AddNodeToSynapseGraph(graph, params.get(), size);
}

OutputShapeInfRetType OpBackend::ComputeOutputShape(at::Stack& stack) {
  m_meta_mode = true;
  AddNode(*m_graph, stack);
  m_meta_mode = false;

  return m_meta;
}

void OpBackend::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const OutputMetaDataVector& output_metadata) {
  m_output_metadata = output_metadata;

  CustomHandler(graph, stack);

  HandleFn(graph, stack);
  HandleInplaceFn(graph, stack);
  HandleOutFn(graph, stack);

  HandleScalarToTensor(graph, stack);
  HandleTypePromotion(graph, stack);
  HandleIntToFloatPromotion(graph, stack);

  AddNode(graph, stack);
}

const synapse_helpers::tensor& OpBackend::CreateShapeTensorInput(
    synapse_helpers::graph& graph,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    synTensorType shape_tensor_type) {
  auto st = habana_helpers::create_shape_tensor(
      GetProxyTensor(dtype, sizes), graph, false, shape_tensor_type);
  m_shape_tensors.emplace_back(std::move(st));
  return m_shape_tensors.back();
}

std::vector<synapse_helpers::tensor> OpBackend::BuildNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    NodeAttr node_attr) {
  if (op->isMetaMode()) {
    auto& meta = op->GetMeta();
    std::vector<synapse_helpers::tensor> out;

    for (const auto& attr : node_attr.output_attrs) {
      const auto& t = GetProxyTensor(attr.dtype, attr.sizes);
      const auto& md = TensorMetaData(
          t.sizes().vec(),
          t.strides().vec(),
          attr.dtype,
          at::MemoryFormat::Contiguous);
      if (attr.final_result_index.has_value()) {
        meta.AddOutputTensor(md);
      } else {
        meta.AddIntermediateTensor(md);
      }
      out.emplace_back(synapse_helpers::tensor::create_placeholder(0, {}, {}));
    }

    return out;
  }

  auto ctx = op->p_context_;
  std::vector<synapse_helpers::tensor> outputs;
  std::vector<synTensor> node_outputs;

  for (const auto& attr : node_attr.output_attrs) {
    if (attr.final_result_index.has_value() and op->IsOutputAvailable()) {
      // HandleOutFn/HandleInplaceFn placed the output in syn_outputs_
      outputs.emplace_back(
          std::move(ctx->syn_outputs_.at(*attr.final_result_index).ref()));
    } else {
      bool is_persistent = attr.final_result_index.has_value() and
          op->m_output_metadata[attr.final_result_index.value()].persistent;
      const auto& t = GetProxyTensor(attr.dtype, attr.sizes);
      bool is_external = attr.final_result_index.has_value() and
          op->m_output_metadata.at(attr.final_result_index.value()).external;
      outputs.emplace_back(
          habana_helpers::is_shape_tensor(attr.tensor_type)
              ? habana_helpers::create_shape_tensor(
                    t, graph, is_persistent, attr.tensor_type)
              : habana_helpers::create_tensor(
                    t, graph, is_persistent, is_external, attr.dtype));

      if (is_persistent) {
        const auto& impl =
            ctx->pt_outputs_.at(*attr.final_result_index).unsafeGetTensorImpl();
        impl->set_sizes_contiguous(attr.sizes);
        impl->set_storage_and_dtype(
            impl->storage(), c10::scalarTypeToTypeMeta(attr.dtype));
      } else if (attr.final_result_index.has_value()) {
        ctx->pt_outputs_.at(*attr.final_result_index) = t;
      }
    }
    node_outputs.emplace_back(outputs.back().get());
  }

  auto input_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      op->kernel_meta_data_.synapse_input_layout);
  auto output_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      op->kernel_meta_data_.synapse_output_layout);

  auto result = graph.add_node(
      std::move(node_attr.inputs),
      std::move(node_outputs),
      node_attr.params,
      node_attr.param_size,
      node_attr.guid,
      nullptr,
      input_layouts.data(),
      output_layouts.data());
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
    c10::optional<int> final_result_index) {
  const auto& guid = "cast_" + habana_helpers::name_suffix_from_type(from) +
      "_to_" + habana_helpers::name_suffix_from_type(to);
  HABANA_ASSERT(from != to, guid, " cannot be used.");

  ns_CastKernel::Params params{};
  SET_CAST_ROUNDING_MODE(guid);
  NodeAttr castnode{
      guid,
      {syn_in},
      {{sizes, to, final_result_index}},
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
    c10::optional<int> final_result_index) {
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

  std::vector<synTensor> input;
  if (!op->isMetaMode() and graph.is_dynamic_graph()) {
    input.emplace_back(op->CreateShapeTensorInput(
                             graph, valtype, constant_outshape, SHAPE_TENSOR)
                           .get());
  }

  auto constant = BuildNode(
      op,
      graph,
      {"constant_" + habana_helpers::name_suffix_from_type(valtype),
       input,
       {{constant_outshape, valtype, final_result_index}},
       &params,
       sizeof(params)});

  return std::move(constant.at(0));
}

synapse_helpers::tensor OpBackend::BuildReshape(
    OpBackend* op,
    synapse_helpers::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  /*
    Inputs:
    * The tensor to reshape : T
        Input Tensor of type T with dimensionality 1-5D.
    * Shape tensor describing output : T
        Input Tensor of type T with dimensionality 1-5D.

    Outputs:
    * The reshaped tensor : T
        Output tensor with the same type as input.

    Types:
    * T : tensor(float32), tensor(bfloat16), tensor(int32)
        A 1D, 2D, 3D, 4D or 5D tensor with the elements of type specified in
        the definition.
  */
  std::vector<synTensor> inputs = {syn_in};
  if (!op->isMetaMode() and graph.is_dynamic_graph()) {
    inputs.emplace_back(
        op->CreateShapeTensorInput(graph, dtype, sizes, SHAPE_TENSOR).get());
  }

  auto reshape = BuildNode(
      op, graph, {"reshape", inputs, {{sizes, dtype, final_result_index}}});
  return std::move(reshape.at(0));
}
} // namespace habana
