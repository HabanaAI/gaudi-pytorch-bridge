/*******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "hpu_ops/op_backend.h"
#include <c10/core/ScalarType.h>
#include "backend/create_pt_tensor.h"
#include "backend/helpers/cast_sequence.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/pt_version_check.h"
#include "habana_kernels/kernel_utils.h"
#include "hpu_ops/hpu_op_helper.h"

namespace sh = synapse_helpers;

namespace {
auto BuildCastGuid(const c10::ScalarType& src, const c10::ScalarType& dst) {
  static const std::string prefix = "cast_";
  const auto srcStr = habana_helpers::name_suffix_from_type(
      src, habana_helpers::isLongTypeSupported(prefix));
  const auto dstStr = habana_helpers::name_suffix_from_type(
      dst, habana_helpers::isLongTypeSupported(prefix));
  const auto guid = prefix + srcStr + "_to_" + dstStr;
  HABANA_ASSERT(
      srcStr != dstStr, guid, " cannot be used, from=", src, " to=", dst);
  return guid;
}
} // namespace

namespace habana {

static at::ScalarType GetScalarType(const at::Stack& stack, int index) {
  const auto& ival = stack.at(index);
  auto type =
      ival.isTensor() ? ival.toTensor().scalar_type() : ival.toScalar().type();

  return habana_helpers::getInternalDtype(type);
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
    : HabanaOperator(
          guid +
          habana_helpers::name_suffix_from_type(
              scalar_type,
              habana_helpers::isLongTypeSupported(guid))),
      m_res_ids{std::move(res_ids)},
      m_inplace_ids{std::move(inplace_ids)},
      m_scalar_ids{std::move(scalar_ids)},
      m_is_outfn{is_outfn},
      m_scalar_type{scalar_type} {
  CreateSynContext(device_id);
}

synTensor OpBackend::syn_in(int index) {
  if (isMetaMode()) {
    return nullptr;
  }

  return SynInput(index).ref().get();
}

sh::tensor& OpBackend::syn_out(int index) {
  if (isMetaMode()) {
    // create dummy tensor with out incrementing tensor id
    static auto ph = sh::tensor::create_placeholder(
        0, {}, {}, false, std::string(), DATA_TENSOR, false);
    return ph;
  }
  return p_context_->syn_outputs_.at(index);
}

sh::tensor_or_ref& OpBackend::SynInput(int index) {
  auto it = syn_inputs_cast_.find(index);
  if (it != syn_inputs_cast_.end()) {
    return it->second;
  }
  return p_context_->syn_inputs_.at(index);
}

OutputMetaDataVector OpBackend::OutputMeta(const at::Stack& stack) const {
  if (m_output_meta_fn) {
    return m_output_meta_fn(stack);
  }
  return {};
}

at::ScalarType OpBackend::HandleDtypePropagation(
    const at::Stack& stack,
    const at::Tensor& t,
    at::ScalarType metadata_dtype) {
  if (c10::ScalarType::Undefined != metadata_dtype) {
    return metadata_dtype;
  }

  auto propagated_dtype = c10::ScalarType::Undefined;

  if (m_promote_type || m_promote_int_to_float) {
    propagated_dtype = habana_helpers::DTypeHelper::get_compute_dtype(
        stack,
        c10::nullopt,
        m_promote_int_to_float ? habana_helpers::DTypeHelper::
                                     DtypePromoteVariant::kPromoteIntToFloat
                               : habana_helpers::DTypeHelper::
                                     DtypePromoteVariant::kPromoteToCommon,
        false,
        c10::nullopt,
        false,
        false);
  }

  if (c10::ScalarType::Undefined == propagated_dtype) {
    propagated_dtype = t.scalar_type();
  }

  HABANA_ASSERT(
      c10::ScalarType::Undefined != propagated_dtype,
      "Unable to find promoted dtype");
  return propagated_dtype;
}

void OpBackend::HandleScalarToTensor(sh::graph& graph, const at::Stack& stack) {
  if (m_scalar_ids.empty()) {
    return;
  }

  for (int m_scalar_id : m_scalar_ids) {
    const at::Scalar& val = stack.at(m_scalar_id).toScalar();
    m_scalar_inputs.emplace(m_scalar_id, val);

    auto constant = ConstantHelper(graph, val);

    if (!isMetaMode()) {
      // Set output from constant as input to this node at index m_scalar_id
      p_context_->syn_inputs_.emplace(
          p_context_->syn_inputs_.cbegin() + m_scalar_id, std::move(constant));
    }
  }
}

void OpBackend::HandleFn(sh::graph& graph, const at::Stack& stack) {
  if (m_res_ids.empty()) {
    return;
  }

  if (!m_output_metadata.empty() &&
      m_output_metadata.at(0).allocated_tensor.has_value()) {
    for (const auto& el : m_output_metadata)
      AllocateSynapseOutput(graph, el.allocated_tensor.value(), el);
    return;
  }

  std::vector<at::Tensor> tensors;
  tensors.reserve(m_output_metadata.size());

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
      tensors.size() == m_output_metadata.size(),
      "Num outputs defined (",
      m_res_ids.size(),
      ") as out_ids is not matching with actual num outputs (",
      m_output_metadata.size());

  int i = 0;
  for (const at::Tensor& t : tensors) {
    const auto& metadata = m_output_metadata.at(i);
    const auto& dtype = HandleDtypePropagation(stack, t, metadata.dtype);

    const auto& output = habana::createPTTensor(
        t, metadata.shape, t.options().dtype(dtype), metadata.persistent);
    AllocateSynapseOutput(graph, output, metadata);
    i++;
  }
}

void OpBackend::HandleOutFn(sh::graph& graph, const at::Stack& stack) {
  if (!m_is_outfn) {
    return;
  }

  // Check Out variant has output shapes else raise exception
  ComputeOutputShapes(stack);

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

void OpBackend::HandleInplaceFn(sh::graph& graph, const at::Stack& stack) {
  if (m_inplace_ids.empty()) {
    return;
  }

  int syn_counter = 0;
  for (int inplace_id : m_inplace_ids) {
    // Index can vary in syn_inputs_ and in stack
    const auto& ival = stack[inplace_id];
    const auto& tensors = ival.isTensor()
        ? static_cast<at::List<at::Tensor>>(ival.toTensor())
        : ival.toTensorList();
    for (auto i = 0u; i < tensors.size(); ++i) {
      p_context_->syn_outputs_.emplace_back(
          habana_helpers::duplicate_tensor_in_memory_section(
              p_context_->syn_inputs_[syn_counter++],
              graph,
              m_output_metadata.at(inplace_id).external));
      p_context_->pt_outputs_.emplace_back(tensors[i]);
    }
  }
}

void OpBackend::HandleTypePromotion(sh::graph& graph, const at::Stack& stack) {
  if (!m_promote_type && !m_promote_int_to_float) {
    return;
  }

  at::Stack op_inputs = stack;
  if (m_is_outfn) {
    op_inputs = {stack.begin(), stack.end() - m_num_out_tensors};
  }
  m_scalar_type = habana_helpers::DTypeHelper::get_compute_dtype(
      op_inputs,
      c10::nullopt,
      m_promote_int_to_float
          ? habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteIntToFloat
          : habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      c10::nullopt,
      false,
      false);

  auto skipScalarCastNeeded = [&](size_t i) -> bool {
    // Scalars (which are not converted to tensors - index not found in
    // m_scalar_ids) do not deliver underlying synTensors, so although they
    // influence result promoted type, they cannot be casted here.
    return stack.at(i).isScalar() &&
        std::find(m_scalar_ids.begin(), m_scalar_ids.end(), i) ==
        m_scalar_ids.end();
  };

  bool cast_inserted = false;
  for (size_t i = 0; i < op_inputs.size(); ++i) {
    if (!(stack[i].isScalar() or stack[i].isTensor())) {
      continue;
    }
    auto input_type = GetScalarType(stack, i);
    if (habana_helpers::pytorch_to_synapse_type(input_type) ==
        habana_helpers::pytorch_to_synapse_type(m_scalar_type)) {
      continue;
    }

    if (skipScalarCastNeeded(i)) {
      continue;
    }

    cast_inserted = true;

    // Insert cast on the input with lower dtype
    auto cast = CastHelper(
        graph,
        syn_in(i),
        stack.at(i).isTensor() ? stack_tensor(stack, i).sizes() : 1,
        input_type,
        m_scalar_type);

    if (!isMetaMode()) {
      // Replace the input with the casted input
      syn_inputs_cast_.emplace(i, std::move(cast));
    }
  }

  if (!cast_inserted) {
    return;
  }

  // Update the guid to reflect the promoted type
  SetGuid(
      guid_.substr(0, guid_.find_last_of('_') + 1) +
      habana_helpers::name_suffix_from_type(m_scalar_type));
}

std::vector<sh::tensor> OpBackend::BuildOp(
    sh::graph& graph,
    const std::string& guid,
    std::vector<synTensor> node_inputs,
    const std::vector<NodeAttr::NodeOutputAttr>& node_output_attr,
    void* params,
    size_t param_size,
    std::string name) {
  return OpBackend::BuildNode(
      this,
      graph,
      {guid,
       std::move(node_inputs),
       node_output_attr,
       params,
       param_size,
       name});
}

sh::tensor OpBackend::CastHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    const at::ScalarType& from,
    const at::ScalarType& to,
    c10::optional<int> final_result_index,
    bool stochastic_rounding_override,
    int sr_seed) {
  return OpBackend::BuildCast(
      this,
      graph,
      syn_in,
      sizes,
      from,
      to,
      final_result_index,
      stochastic_rounding_override,
      sr_seed);
}

sh::tensor OpBackend::ConstantHelper(
    sh::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    const at::IntArrayRef constant_outshape,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildConstant(
      this, graph, val, force_type, constant_outshape, final_result_index);
}

sh::tensor OpBackend::BroadcastHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildBroadcast(
      this, graph, syn_in, sizes, dtype, final_result_index);
}

sh::tensor OpBackend::ReshapeHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildReshape(
      this, graph, syn_in, sizes, dtype, final_result_index);
}

sh::tensor OpBackend::PermuteHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::IntArrayRef permutation,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildPermute(
      this, graph, syn_in, sizes, permutation, dtype, final_result_index);
}

sh::tensor OpBackend::IdentityHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildIdentity(
      this, graph, syn_in, sizes, dtype, final_result_index);
}

void OpBackend::AddNode(sh::graph& graph, const at::Stack& stack) {
  if (isMetaMode()) {
    if (m_is_outfn) { // out place fn
      for (int i = m_num_out_tensors; i > 0; --i) {
        const auto& t = stack.at(stack.size() - i).toTensor();
        m_meta.AddOutputTensor(TensorMetaData(
            t.sizes().vec(),
            t.strides().vec(),
            t.scalar_type(),
            t.suggest_memory_format()));
      }
    } else if (!m_inplace_ids.empty()) { // in place fn
      for (int inplace_id : m_inplace_ids) {
        // Index can vary in syn_inputs_ and in stack
        const auto& ival = stack.at(inplace_id);
        const auto& tensors = ival.isTensor()
            ? static_cast<at::List<at::Tensor>>(ival.toTensor())
            : ival.toTensorList();
        for (const at::Tensor& tensor : tensors) {
          m_meta.AddOutputTensor(TensorMetaData(
              tensor.sizes().vec(),
              tensor.strides().vec(),
              tensor.scalar_type(),
              tensor.suggest_memory_format()));
        }
      }
    } else { // normal fn
      const auto& outshapes = ComputeOutputShapes(stack);
      for (int res_id : m_res_ids) {
        // Index can vary in syn_inputs_ and in stack
        const auto& ival = stack.at(res_id);
        const auto& tensors = ival.isTensor()
            ? static_cast<at::List<at::Tensor>>(ival.toTensor())
            : ival.toTensorList();
        for (auto i = 0u; i < tensors.size(); ++i) {
          const auto& outshape =
              outshapes.empty() ? tensors[i].sizes() : outshapes[i];
          const auto& strides = HabanaOperator::CalculateStrides(
              outshape.vec(), at::MemoryFormat::Contiguous);
          m_meta.AddOutputTensor(TensorMetaData(
              outshape.vec(),
              strides,
              tensors[i].scalar_type(),
              tensors[i].suggest_memory_format()));
        }
      }
    }
    return;
  }
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  AddNodeToSynapseGraph(graph, params.get(), size);
}

OutputShapeInfRetType OpBackend::ComputeOutputShape(at::Stack& stack) {
  m_meta_mode = true;
  auto& device = sh::HPURegistrar::get_device(0);
  auto graph = absl::get<sh::graph>(sh::graph::create(device, {}, true));

  HandleScalarToTensor(graph, stack);

  if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
    HandleTypePromotion(graph, stack);
  }

  AddNode(graph, stack);
  m_meta_mode = false;

  return m_meta;
}

void OpBackend::PopulateMetadata(
    const at::Stack& stack,
    const OutputMetaDataVector& output_metadata) {
  m_output_metadata = output_metadata;

  if (UsesOutputMeta()) {
    const auto& meta = OutputMeta(stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(meta.size() == m_output_metadata.size());
    for (int i = 0; i < m_output_metadata.size(); ++i) {
      m_output_metadata[i].shape = meta[i].shape;
      m_output_metadata[i].dtype = meta[i].dtype;
      m_output_metadata[i].strides = meta[i].strides;
      m_output_metadata[i].mem_format = meta[i].mem_format;
    }
  } else if (m_res_ids.size()) {
    auto outshapes = ComputeOutputShapes(stack);
    if (outshapes.empty()) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          m_res_ids.size() == m_output_metadata.size());
      for (int res_id : m_res_ids) {
        outshapes.emplace_back(stack_tensor(stack, res_id).sizes().vec());
      }
    }
    for (int i = 0; i < m_output_metadata.size(); ++i) {
      m_output_metadata[i].shape = outshapes[i];
    }
  }
}

void OpBackend::AllocateAndAddSynapseNode(
    sh::graph& graph,
    at::Stack& stack,
    const OutputMetaDataVector& output_metadata) {
  PopulateMetadata(stack, output_metadata);

  CustomHandler(graph, stack);

  HandleFn(graph, stack);
  HandleInplaceFn(graph, stack);
  HandleOutFn(graph, stack);

  HandleScalarToTensor(graph, stack);

  if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
    HandleTypePromotion(graph, stack);
  }

  AddNode(graph, stack);
}

void OpBackend::CreateShapeTensorInput(
    sh::graph& graph,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    std::vector<synTensor>& inputs,
    synTensorType shape_tensor_type) {
  // Add intermediate shape tensor
  if (isMetaMode()) {
    auto& meta = GetMeta();
    const auto& st = GetProxyTensor(dtype, sizes);
    const auto& md = TensorMetaData(
        st.sizes().vec(),
        st.strides().vec(),
        dtype,
        at::MemoryFormat::Contiguous);
    meta.AddShapeTensor(md);

    return;
  }

  if (graph.is_dynamic_graph()) {
    auto st = habana_helpers::create_shape_tensor(
        GetProxyTensor(dtype, sizes), graph, false, shape_tensor_type);
    st.set_intermediate_shape_tensor();
    m_shape_tensors.emplace_back(std::move(st));
    inputs.emplace_back(m_shape_tensors.back().get());
  }
}

std::vector<sh::tensor> OpBackend::BuildNode(
    OpBackend* op,
    sh::graph& graph,
    NodeAttr node_attr) {
  if (op->isMetaMode()) {
    auto& meta = op->GetMeta();
    const auto& output_attrs_size = node_attr.output_attrs.size();
    std::vector<sh::tensor> out;
    out.reserve(output_attrs_size);

    for (const auto& attr : node_attr.output_attrs) {
      const auto& attr_strides = HabanaOperator::CalculateStrides(
          attr.sizes.vec(), at::MemoryFormat::Contiguous);
      const auto& md = TensorMetaData(
          attr.sizes.vec(),
          attr_strides,
          attr.dtype,
          at::MemoryFormat::Contiguous);

      if (habana_helpers::is_shape_tensor(attr.tensor_type)) {
        meta.AddShapeTensor(md);
      }
      // AddShapeTensor call is independent of AddOutputTensor and
      // AddIntermediateOutputTensor. That is why no else if.
      if (attr.final_result_index.has_value() ||
          attr.inplace_out_ptr.has_value()) {
        meta.AddOutputTensor(md);
      } else {
        meta.AddIntermediateTensor(md);
      }

      // create dummy tensor with only sizes and strides info
      out.emplace_back(
          sh::tensor::create_placeholder(attr.sizes.vec(), attr_strides));
    }

    return out;
  }

  const auto& ctx = op->p_context_;
  std::vector<sh::tensor> outputs;
  outputs.reserve(node_attr.output_attrs.size());
  std::vector<synTensor> node_outputs;
  node_outputs.reserve(node_attr.output_attrs.size());

  for (const auto& attr : node_attr.output_attrs) {
    bool is_final_result = attr.final_result_index.has_value();
    if (is_final_result and
        (op->IsOutputAvailable() or op->UsesOutputMeta() or
         op->GetOutputMetaData(*attr.final_result_index)
             .allocated_tensor.has_value())) {
      // - HandleOutFn/HandleInplaceFn placed the output(s) in syn_outputs_
      // - HandleFn placed the output(s) in in syn_outputs_ when the op uses
      // output_meta
      outputs.emplace_back(
          std::move(ctx->syn_outputs_.at(*attr.final_result_index).ref()));
    } else if (attr.inplace_out_ptr) {
      if (std::holds_alternative<sh::tensor*>(*attr.inplace_out_ptr)) {
        outputs.emplace_back(habana_helpers::duplicate_tensor_in_memory_section(
            *(std::get<sh::tensor*>(*attr.inplace_out_ptr)),
            graph,
            /* is_external */ false));
      } else {
        outputs.emplace_back(habana_helpers::duplicate_tensor_in_memory_section(
            op->SynInput(std::get<int>(*attr.inplace_out_ptr)),
            graph,
            /* is_external */ false));
      }
    } else {
      bool is_persistent = false;
      bool is_external = false;

      if (is_final_result) {
        const auto& metadata = op->GetOutputMetaData(*attr.final_result_index);
        is_persistent = metadata.persistent;
        is_external = metadata.external;
      }

      const auto& t = GetProxyTensor(attr.dtype, attr.sizes);
      outputs.emplace_back(
          habana_helpers::is_shape_tensor(attr.tensor_type)
              ? habana_helpers::create_shape_tensor(
                    t, graph, is_persistent, attr.tensor_type)
              : attr.syn_data_type == syn_type_na
                  ? habana_helpers::create_tensor(
                        t,
                        graph,
                        is_persistent,
                        is_external,
                        attr.dtype,
                        node_attr.inf_name,
                        node_attr.inf_name)
                  : habana_helpers::create_tensor(
                        t,
                        graph,
                        is_persistent,
                        is_external,
                        attr.syn_data_type,
                        node_attr.inf_name,
                        node_attr.inf_name));

      if (is_persistent) {
        const auto& impl =
            ctx->pt_outputs_.at(*attr.final_result_index).unsafeGetTensorImpl();
        // Free the old storage
        impl->FreeMemory();

        auto storage = c10::make_intrusive<c10::StorageImpl>(
            c10::StorageImpl::use_byte_size_t(),
            c10::multiply_integers(attr.sizes) *
                c10::scalarTypeToTypeMeta(attr.dtype).itemsize(),
            habana::getHABANADeviceAllocator(),
            true);
        impl->set_storage_and_dtype(
            storage, c10::scalarTypeToTypeMeta(attr.dtype));
        impl->set_sizes_contiguous(attr.sizes);

      } else if (is_final_result) {
        ctx->pt_outputs_.at(*attr.final_result_index) = t;
      }
    }
    node_outputs.emplace_back(outputs.back().get());
  }

  auto input_layouts = sh::layouts::getSynapseLayoutFormat(
      op->kernel_meta_data_.synapse_input_layout);
  auto output_layouts = sh::layouts::getSynapseLayoutFormat(
      op->kernel_meta_data_.synapse_output_layout);

  HABANA_ASSERT(
      input_layouts.empty() || input_layouts.size() >= node_attr.inputs.size(),
      "Missing layouts for synapse inputs");
  HABANA_ASSERT(
      output_layouts.empty() || output_layouts.size() >= node_outputs.size(),
      "Missing layouts for synapse outputs");

  auto result = graph.add_node(
      std::move(node_attr.inputs),
      std::move(node_outputs),
      node_attr.params,
      node_attr.param_size,
      node_attr.guid,
      nullptr,
      input_layouts.empty() ? nullptr : input_layouts.data(),
      output_layouts.empty() ? nullptr : output_layouts.data(),
      op->deterministic);

  HABANA_ASSERT(
      ok(result),
      "Adding ",
      node_attr.guid,
      " to graph failed with ",
      get_error(result).error,
      Logger::synStatusToStr(get_error(result).status));

  return outputs;
}

sh::tensor OpBackend::BuildCast(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    const at::IntArrayRef sizes,
    const at::ScalarType& from,
    const at::ScalarType& to,
    c10::optional<int> final_result_index,
    bool stochastic_rounding_override,
    int sr_seed) {
  // Verify from and to types correctness
  BuildCastGuid(from, to);

  // We want either 1 or 0 as results and not the entire i8 range as a bool
  // output.
  if (to == at::kBool) {
    auto zero_tensor = OpBackend::BuildConstant(op, graph, 0, from);

    auto eq = OpBackend::BuildNode(
        op,
        graph,
        {"equal_fwd_" + habana_helpers::name_suffix_from_type(from),
         {syn_in, zero_tensor.get()},
         {{sizes, c10::ScalarType::Bool}}});

    auto ne = OpBackend::BuildNode(
        op,
        graph,
        {"not_fwd_i8",
         {eq[0].get()},
         {{sizes, c10::ScalarType::Bool, final_result_index}}});
    return std::move(ne[0]);
  }

  habana_helpers::CastTypes cast_types{
      habana_helpers::DataTypeToCastType(from),
      habana_helpers::DataTypeToCastType(to)};

  const auto cast_sequence = habana_helpers::get_cast_sequence(cast_types);

  synTensor* input = &syn_in;
  std::vector<sh::tensor> casts;
  casts.reserve(cast_sequence.size());
  for (size_t i = 0; i < cast_sequence.size(); ++i) {
    const auto src =
        habana_helpers::CastTypeToDataType(cast_sequence.at(i).from_);
    const auto dst =
        habana_helpers::CastTypeToDataType(cast_sequence.at(i).to_);
    const auto cast_guid = BuildCastGuid(src, dst);

    c10::variant<ns_CastKernel::Params, ns_CastKernel::ParamsV2> params;

#if HAVE_FP8R152_SUPPORT
    bool use_explicit_seed = (0 != sr_seed) && to == at::kFp8r152;
#else
    bool use_explicit_seed = false;
#endif

    if (use_explicit_seed) {
      // Usage of ParamsV2 type induces explicit seed mode in TPC
      params.emplace<ns_CastKernel::ParamsV2>();
      c10::get<ns_CastKernel::ParamsV2>(params).seed = sr_seed;
    } else {
      params.emplace<ns_CastKernel::Params>();
    }

    void* params_ptr = c10::visit(
        [to, stochastic_rounding_override](auto& var) {
          var.round_mode = habana_helpers::get_cast_rounding_mode(
              to, stochastic_rounding_override);
          return reinterpret_cast<void*>(&var);
        },
        params);
    size_t params_size =
        c10::visit([](const auto& var) { return sizeof(var); }, params);

    auto is_last = (i + 1) == cast_sequence.size();
    auto output_index = is_last ? final_result_index : c10::nullopt;
    NodeAttr castnode{
        cast_guid,
        {*input},
        {{sizes, dst, output_index}},
        params_ptr,
        params_size};
    auto cast = BuildNode(op, graph, std::move(castnode));
    casts.emplace_back(std::move(cast.at(0)));
    input = &casts.back().get();
  }

  HABANA_ASSERT(!casts.empty(), "Empty vector of casts.");
  return std::move(casts.back());
}

sh::tensor OpBackend::BuildConstant(
    OpBackend* op,
    sh::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    const at::IntArrayRef constant_outshape,
    c10::optional<int> final_result_index) {
  // For lazy eager mode, Allocate constant synapse tensor
  // for non-persistent tensor of size {1}.
  // To do: Check support for force data type to const tensor.
  //        Confirm if CGUID dtype promotion can take care of it.
  if (op->GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER &&
      !final_result_index.has_value() && !force_type.has_value() &&
      constant_outshape.equals({1})) {
    return OpBackend::BuildConstantTensor(op, graph, val);
  }

  at::ScalarType valtype =
      force_type.has_value() ? force_type.value() : val.type();

  ns_ConstantKernel::Params params{};
  if (valtype == c10::ScalarType::Int or valtype == c10::ScalarType::Long) {
    get<int>(params.constant) = val.to<int>();
    if (habana_helpers::is_downcast_to_int_needed(valtype)) {
      valtype = c10::ScalarType::Int;
    }
  } else {
    get<float>(params.constant) = val.to<float>();
  }

  std::vector<synTensor> input;
  op->CreateShapeTensorInput(graph, valtype, constant_outshape, input);

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

sh::tensor OpBackend::BuildConstantTensor(
    OpBackend* op,
    sh::graph& graph,
    const at::Scalar& val,
    [[maybe_unused]] const at::IntArrayRef outshape) {
  if (op->isMetaMode()) {
    // dummy synapse tensor
    return sh::tensor::create_placeholder({1}, {1});
  }

  return op->AllocateConstantSynapseTensor(graph, val);
}

sh::tensor OpBackend::BuildBroadcast(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  std::vector<synTensor> inputs = {syn_in};
  op->CreateShapeTensorInput(graph, dtype, sizes, inputs);

  auto broadcast = BuildNode(
      op,
      graph,
      {"broadcast_" + habana_helpers::name_suffix_from_type(dtype),
       inputs,
       {{sizes, dtype, final_result_index}}});
  return std::move(broadcast.at(0));
}

sh::tensor OpBackend::BuildPermute(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::IntArrayRef permutation,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  std::vector<synTensor> inputs = {syn_in};

  int dims_number = sizes.size();

  synTransposeParamsNDims params;
  params.tensorDim = dims_number;
  // params.permute has to be populated in a reverse order for HPU FCD-LCD order
  for (int i = 0; i < dims_number; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(
        dims_number - permutation[permutation.size() - i - 1] - 1);
  }
  for (int i = dims_number; i < HABANA_DIM_MAX; i++) {
    params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }

  auto compute_output_shape = [](at::IntArrayRef self_sizes,
                                 at::IntArrayRef permutation) {
    TORCH_CHECK(
        self_sizes.size() == permutation.size(),
        "Number of dims in tensor don't match in permutation");
    auto new_sizes = self_sizes.vec();
    new_sizes[new_sizes.size() - 1] =
        self_sizes[permutation[new_sizes.size() - 1]];
    for (int i = new_sizes.size() - 2; i >= 0; i--) {
      new_sizes[i] = self_sizes[permutation[i]];
    }
    return new_sizes;
  };

  auto permute = BuildNode(
      op,
      graph,
      {"transpose",
       inputs,
       {{std::move(compute_output_shape(sizes, permutation)),
         dtype,
         final_result_index}},
       &params,
       sizeof(params)});
  return std::move(permute.at(0));
}

sh::tensor OpBackend::BuildReshape(
    OpBackend* op,
    sh::graph& graph,
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
  op->CreateShapeTensorInput(graph, dtype, sizes, inputs);

  auto reshape = BuildNode(
      op, graph, {"reshape", inputs, {{sizes, dtype, final_result_index}}});
  return std::move(reshape.at(0));
}

sh::tensor OpBackend::BuildIdentity(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  auto identity = BuildNode(
      op, graph, {"identity", {syn_in}, {{sizes, dtype, final_result_index}}});
  return std::move(identity.at(0));
}

std::vector<sh::tensor> OpBackend::BuildNonZero(
    OpBackend* op,
    sh::graph& graph,
    sh::tensor& inTensor,
    at::IntArrayRef outShape,
    at::ScalarType inScalarType,
    c10::optional<int> finalResultIndex) {
  constexpr auto shapeTensorDim = 5;
  constexpr auto outShapeAllowedRank = 2;

  HABANA_ASSERT(
      outShape.size() == outShapeAllowedRank,
      "Non-zero output tensor rank is always ",
      outShapeAllowedRank,
      " here ",
      outShape.size(),
      " rank was given");

  const std::string guid =
      "non_zero_fwd_" + habana_helpers::name_suffix_from_type(inScalarType);

  return op->BuildOp(
      graph,
      guid,
      {inTensor.get()},
      {NodeAttr::NodeOutputAttr{outShape, at::kInt, finalResultIndex},
       NodeAttr::NodeOutputAttr{
           {shapeTensorDim}, at::kInt, c10::nullopt, DEVICE_SHAPE_TENSOR}});
}

sh::tensor OpBackend::BuildScatterNDOnnx(
    OpBackend* op,
    sh::graph& graph,
    const std::vector<synTensor>& inTensors,
    at::IntArrayRef outShape,
    at::ScalarType inScalarType,
    int validCountTensorRank,
    c10::optional<int> finalResultIndex) {
  const auto& inputTensor = inTensors[0];
  const auto& indexTensor = inTensors[1];
  const auto& updatesTensor = inTensors[2];

  constexpr auto allowedNrOfInTensors = 3; // +1 optional
  constexpr auto allowedValidCountTensorRank = 1;

  const auto nrOfInTensors = inTensors.size();

  HABANA_ASSERT(
      nrOfInTensors == allowedNrOfInTensors or
          nrOfInTensors == allowedNrOfInTensors + 1,
      "ScatterND input must have ",
      allowedNrOfInTensors,
      " or ",
      allowedNrOfInTensors + 1,
      " tensors "
      " here ",
      nrOfInTensors,
      " tensors was given");
  if (nrOfInTensors == allowedNrOfInTensors + 1) {
    HABANA_ASSERT(
        validCountTensorRank == allowedValidCountTensorRank,
        "ScatterND ValidCount tensor must have rank 1");
  }

  const std::string guid = "scatter_nd_onnx_fwd_" +
      habana_helpers::name_suffix_from_type(inScalarType);

  return std::move(op->BuildOp(
                         graph,
                         guid,
                         [&]() {
                           const auto& validCountTensor = inTensors[3];
                           std::vector<synTensor> res;
                           res.reserve(4);
                           res.insert(
                               res.begin(),
                               {inputTensor, indexTensor, updatesTensor});

                           if (nrOfInTensors == allowedNrOfInTensors + 1) {
                             res.push_back(validCountTensor);
                           }

                           return res;
                         }(),
                         {NodeAttr::NodeOutputAttr{
                             outShape, inScalarType, finalResultIndex}})
                       .at(0));
}
} // namespace habana
