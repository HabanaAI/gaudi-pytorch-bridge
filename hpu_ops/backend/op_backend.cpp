/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/cast_sequence.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/runtime_config.h"
#include "backend/helpers/tensor_utils.h"
#include "common/utils.h"
#include "habana_helpers/dtype_helpers.h"
#include "habana_helpers/pt_version_check.h"
#include "habana_kernels/kernel_utils.h"
#include "hpu_ops/common/scalar_dtype_range.h"
#include "hpu_ops/hpu_op_helper.h"

namespace sh = synapse_helpers;

namespace {

auto GetPrecisionString(
    const c10::ScalarType& dtype,
    const std::string_view& prefix) {
  return synapse_helpers::graph::name_suffix_from_type(
      habana_helpers::pytorch_to_synapse_type(dtype),
      habana_helpers::isLongTypeSupported(prefix));
}

auto BuildCastGuid(const c10::ScalarType& src, const c10::ScalarType& dst) {
  using namespace std::literals;
  static const std::string_view prefix = "cast_"sv;

  const auto srcStr = GetPrecisionString(src, prefix);
  const auto dstStr = GetPrecisionString(dst, prefix);
  const auto guid =
      std::string{prefix}.append(srcStr).append("_to_"sv).append(dstStr);
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

  // return Bool instead of Char for handling copy from/to Bool in Cast Node
  if (type == at::kBool)
    return type;

  return habana_helpers::getInternalDtype(type);
}

static at::Tensor GetProxyTensor(
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    c10::optional<unsigned> exp_bias = c10::nullopt) {
  const auto& t = at::detail::make_tensor<c10::TensorImpl>(
      c10::DispatchKeySet{at::DispatchKey::HPU, at::DispatchKey::AutogradHPU},
      c10::scalarTypeToTypeMeta(dtype),
      c10::Device(c10::kHPU, 0));
  t.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  habana_helpers::set_tensor_exp_bias(t, exp_bias);

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
    : HabanaOperator(get_guid_with_precision(
          guid,
          scalar_type,
          habana_helpers::isLongTypeSupported(guid))),
      m_res_ids{std::move(res_ids)},
      m_inplace_ids{std::move(inplace_ids)},
      m_scalar_ids{std::move(scalar_ids)},
      m_is_outfn{is_outfn},
      m_scalar_type{scalar_type} {
  CreateSynContext(device_id);
}

synTensor OpBackend::syn_in(size_t index) {
  if (isOutputInfMode()) {
    return nullptr;
  }

  return SynInput(index).ref().get();
}

sh::tensor& OpBackend::syn_out(size_t index) {
  if (isOutputInfMode()) {
    // create dummy tensor with out incrementing tensor id
    static auto ph = sh::tensor::create_placeholder(
        0, {}, {}, false, std::string(), DATA_TENSOR, false);
    return ph;
  }
  return p_context_->syn_outputs_.at(index);
}

synTensor OpBackend::syn_seed() {
  if (isOutputInfMode()) {
    return nullptr;
  }

  HABANA_ASSERT(p_context_->syn_seed_.has_value(), "seed is not populated");
  return p_context_->syn_seed_.value().ref().get();
}

sh::tensor_or_ref& OpBackend::SynInput(size_t index) {
  auto it = syn_inputs_cast_.find(index);
  if (it != syn_inputs_cast_.end()) {
    return it->second;
  }
  return p_context_->syn_inputs_.at(index);
}

const sh::tensor_or_ref& OpBackend::ReadSynInput(size_t index) {
  return SynInput(index);
}

void OpBackend::EraseSynInput(int index) {
  if (p_context_->syn_inputs_.begin() + index !=
      p_context_->syn_inputs_.end()) {
    p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin() + index);
  }
}

OutputMetaDataVector OpBackend::OutputMeta(const at::Stack& stack) const {
  if (m_output_meta_fn) {
    return m_output_meta_fn(stack);
  } else if (m_partial_output_meta_fn) {
    OutputMetaDataVector meta;
    for (const auto& m : m_partial_output_meta_fn(stack)) {
      meta.emplace_back(m.dtype, m.shape);
    }
    return meta;
  }
  return {};
}

void OpBackend::HandleScalarToTensorSTMeta(
    habana_helpers::IShapeList& inputs) const {
  if (m_scalar_ids.empty()) {
    return;
  }
  PT_BRIDGE_DEBUG("HandleScalarToTensorSTMeta m_scalar_ids:", m_scalar_ids);
  static_cast<void>(inputs);
  for (int m_scalar_id : m_scalar_ids) {
    static_cast<void>(m_scalar_id);
    std::vector<int64_t> out_shape = {1};
    PT_BRIDGE_DEBUG("HandleScalarToTensorSTMeta constant shape ", out_shape);
    habana_helpers::UpdateSTShapeInfo(out_shape);
  }
}

bool OpBackend::STMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) const {
  if (m_st_meta_fn) {
    HandleScalarToTensorSTMeta(inputs);
    return m_st_meta_fn(inputs, outputs);
  } else {
    // Return false for failure case
    PT_BRIDGE_DEBUG("ST meta not registered !!!");
    return false;
  }
}

void OpBackend::HandleScalarToTensor(sh::graph& graph, const at::Stack& stack) {
  if (m_scalar_ids.empty()) {
    return;
  }
  for (int m_scalar_id : m_scalar_ids) {
    const at::Scalar& val = stack.at(m_scalar_id).toScalar();
    auto constant = ConstantHelper(graph, val);
    if (!isOutputInfMode()) {
      // Set output from constant as input to this node at index m_scalar_id
      p_context_->syn_inputs_.emplace(
          p_context_->syn_inputs_.cbegin() + m_scalar_id, std::move(constant));
    }
  }
}
void OpBackend::HandleFn(sh::graph& graph) {
  if (m_res_ids.empty()) {
    return;
  }

  for (const auto& metadata : m_output_metadata) {
    if (!graph.is_dry_run() && metadata.allocated_tensor.has_value()) {
      const auto output = metadata.allocated_tensor.value();
      if (metadata.exp_bias) {
        habana_helpers::set_tensor_exp_bias(output, metadata.exp_bias);
      }
      AllocateSynapseOutput(graph, output, metadata);
    } else {
      const auto t = GetProxyTensor(metadata.dtype, metadata.shape);
      const auto& output = metadata.strides.empty()
          ? habana::createPTTensor(
                t, metadata.shape, t.options(), metadata.persistent)
          : habana::createPTTensor(
                t,
                metadata.shape,
                metadata.strides,
                t.options(),
                metadata.mem_format,
                metadata.persistent);
      if (metadata.exp_bias) {
        habana_helpers::set_tensor_exp_bias(output, metadata.exp_bias);
      }
      AllocateSynapseOutput(graph, output, metadata);
    }
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
  int out_counter = 0;
  for (size_t stack_id = 0, inplace_ids_pos = 0;
       (stack_id < stack.size()) && (inplace_ids_pos < m_inplace_ids.size());
       ++stack_id) {
    const auto& ival = stack[stack_id];
    const auto& tensors = ival.isTensor()
        ? static_cast<at::List<at::Tensor>>(ival.toTensor())
        : ival.isTensorList() ? ival.toTensorList() : at::List<at::Tensor>{};

    const auto inplace_id = m_inplace_ids[inplace_ids_pos];
    if (inplace_id != (int)stack_id) {
      syn_counter += tensors.size();
    } else {
      for (auto i = 0u; i < tensors.size(); ++i) {
        p_context_->syn_outputs_.emplace_back(
            habana_helpers::duplicate_tensor_in_memory_section(
                p_context_->syn_inputs_[syn_counter++],
                graph,
                m_output_metadata.at(out_counter++).external));
        p_context_->pt_outputs_.emplace_back(tensors[i]);
      }
      ++inplace_ids_pos;
    }
  }
}

static c10::optional<c10::ScalarType> get_dtype_for_large_scalar(
    const std::string& guid_,
    const at::Stack& stack) {
  if ((guid_.find("mult") == std::string::npos &&
       guid_.find("div") == std::string::npos) ||
      stack.size() < 2 || !stack.at(0).isTensor() || !stack.at(1).isScalar()) {
    return c10::nullopt;
  }

  const c10::ScalarType self_type = stack.at(0).toTensor().scalar_type();
  const float value = stack.at(1).toScalar().toFloat();
  c10::optional<c10::ScalarType> dtype = c10::nullopt;

  if (is_value_out_of_scalar_range(value, self_type)) {
    dtype = torch::kFloat;
  }

  return dtype;
}

void OpBackend::HandleTypePromotion(sh::graph& graph, const at::Stack& stack) {
  if (!m_promote_type && !m_promote_int_to_float) {
    return;
  }

  at::Stack op_inputs = stack;
  if (m_is_outfn) {
    op_inputs = {stack.begin(), stack.end() - m_num_out_tensors};
  }

  // In case of mul/div with scalars out of dtype range we need to perform
  // computation in fp32
  c10::optional<c10::ScalarType> dtype =
      get_dtype_for_large_scalar(guid_, stack);
  m_scalar_type = habana_helpers::DTypeHelper::get_compute_dtype(
      op_inputs,
      c10::nullopt,
      m_promote_int_to_float
          ? habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteIntToFloat
          : habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      dtype,
      false,
      false);

  // For comparison op, bool inputs need to be casted to uint8.
  if (m_cast_bool_to_uint8 && m_scalar_type == at::kBool) {
    m_scalar_type = at::kByte;
  }

  auto skipScalarCastNeeded = [&](size_t i) -> bool {
    // Scalars (which are not converted to tensors - index not found in
    // m_scalar_ids) do not deliver underlying synTensors, so although they
    // influence result promoted type, they cannot be casted here.
    return stack.at(i).isScalar() &&
        std::find(m_scalar_ids.begin(), m_scalar_ids.end(), i) ==
        m_scalar_ids.end();
  };

  auto isEmptyOptionalInput = [&](size_t i) -> bool {
    // Empty optional inputs do not deliver underlying synTensors
    return stack.at(i).isNone();
  };

  auto isNotScalarAndNotTensor = [&](size_t i) -> bool {
    return !(stack.at(i).isScalar() or stack.at(i).isTensor());
  };

  bool cast_inserted = false;

  for (auto [i, offset] = std::tuple<size_t, size_t>{0, 0};
       i < op_inputs.size();
       ++i) {
    if (skipScalarCastNeeded(i) or isEmptyOptionalInput(i) or
        isNotScalarAndNotTensor(i)) {
      // When stack input doesn't deliver underlying synTensor, there is a shift
      // in indexing of inputs from stack and from synTensors which is adjusted
      // by offest value
      ++offset;
      continue;
    }

    auto input_type = GetScalarType(stack, i);
    if (habana_helpers::pytorch_to_synapse_type(input_type) ==
        habana_helpers::pytorch_to_synapse_type(m_scalar_type)) {
      continue;
    }

    cast_inserted = true;

    // Insert cast on the input with lower dtype
    auto cast = BuildCast(
        this,
        graph,
        syn_in(i - offset),
        stack.at(i).isTensor() ? stack_tensor(stack, i).sizes() : 1,
        input_type,
        m_scalar_type);

    if (!isOutputInfMode()) {
      // Replace the input with the casted input
      syn_inputs_cast_.emplace(i, std::move(cast));
    }
  }

  if (!cast_inserted) {
    return;
  }

  // Update the guid to apply truncation mode
  update_guid_trunc_mode(guid_, m_scalar_type);
  // Update the guid to reflect the promoted type
  update_guid_dtype(guid_, m_scalar_type);
}

void OpBackend::HandleHwScaling(const at::Stack& stack, const size_t i) {
  if (m_hw_scaling_ids.size() <= i or m_hw_scaling_ids[i] == -1) {
    return;
  }

  const auto& input = stack[m_hw_scaling_ids[i]];
  HABANA_ASSERT(input.isTensor());
  const auto& input_t = input.toTensor();
  if (input_t.scalar_type() != at::ScalarType::Float8_e5m2 &&
      input_t.scalar_type() != at::ScalarType::Float8_e4m3fn) {
    return;
  }

  m_output_metadata[i].exp_bias = habana_helpers::get_tensor_exp_bias(input_t);
}

std::vector<sh::tensor> OpBackend::BuildOp(
    sh::graph& graph,
    std::string guid,
    std::vector<synTensor>&& node_inputs,
    std::vector<NodeAttr::NodeOutputAttr> node_output_attr,
    void* params,
    size_t param_size,
    std::string name) {
  return OpBackend::BuildNode(
      this,
      graph,
      {std::move(guid),
       std::move(node_inputs),
       std::move(node_output_attr),
       params,
       param_size,
       std::move(name)});
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
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
  return OpBackend::BuildBroadcast(
      this, graph, syn_in, sizes, dtype, final_result_index, exp_bias);
}

sh::tensor OpBackend::ReshapeHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
  return OpBackend::BuildReshape(
      this, graph, syn_in, sizes, dtype, final_result_index, exp_bias);
}

sh::tensor OpBackend::PermuteHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::IntArrayRef permutation,
    at::ScalarType dtype,
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
  return OpBackend::BuildPermute(
      this,
      graph,
      syn_in,
      sizes,
      permutation,
      dtype,
      final_result_index,
      exp_bias);
}

sh::tensor OpBackend::IdentityHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
  return OpBackend::BuildIdentity(
      this, graph, syn_in, sizes, dtype, final_result_index, exp_bias);
}

sh::tensor OpBackend::SqueezeHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<unsigned> axis,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildSqueeze(
      this, graph, syn_in, sizes, dtype, axis, final_result_index);
}

sh::tensor OpBackend::ExpandDimsHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    unsigned axis,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildExpandDims(
      this, graph, syn_in, sizes, dtype, axis, final_result_index);
}

sh::tensor OpBackend::FlattenHelper(
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  return OpBackend::BuildFlatten(
      this, graph, syn_in, sizes, dtype, final_result_index);
}

void OpBackend::AddNode(sh::graph& graph, const at::Stack& stack) {
  m_num_syn_nodes++;
  if (isOutputInfMode()) {
    if (m_is_outfn) { // out place fn
      for (int i = m_num_out_tensors; i > 0; --i) {
        const auto& t = stack.at(stack.size() - i).toTensor();
        m_output_inf_meta.AddOutputTensor(TensorMetaData(
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
          m_output_inf_meta.AddOutputTensor(TensorMetaData(
              tensor.sizes().vec(),
              tensor.strides().vec(),
              tensor.scalar_type(),
              tensor.suggest_memory_format()));
        }
      }
    } else { // normal fn
      if (UsesOutputMeta()) {
        auto meta = OutputMeta(stack);
        for (const auto& metadata : meta) {
          m_output_inf_meta.AddOutputTensor(TensorMetaData(
              metadata.shape,
              HabanaOperator::CalculateStrides(
                  metadata.shape, at::MemoryFormat::Contiguous),
              metadata.dtype,
              at::MemoryFormat::Contiguous));
        }
      } else {
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
            m_output_inf_meta.AddOutputTensor(TensorMetaData(
                outshape.vec(),
                strides,
                tensors[i].scalar_type(),
                tensors[i].suggest_memory_format()));
          }
        }
      }
    }
    size_t size = 0;
    auto params = FillParams(stack, size);
    // populate node params
    PT_BRIDGE_DEBUG(
        "OpBackend adding params data=", params.get(), ", params size=", size);

    m_output_inf_meta.AddNodeParams(params.get(), size);
    return;
  }
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  AddNodeToSynapseGraph(graph, params.get(), size);
}

InferOutputMetaRetType OpBackend::InferOutputMeta(at::Stack& stack) {
  m_output_inf_mode = true;
  auto& device = habana::HPURegistrar::get_device(0).syn_device();
  auto graph = sh::graph::create(device, {}, true);

  PopulateMetadata(stack, GetOutputMetaData());

  HandleScalarToTensor(graph, stack);

  if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
    HandleTypePromotion(graph, stack);
  }

  AddNode(graph, stack);
  m_output_inf_mode = false;

  return m_output_inf_meta;
}

void OpBackend::PopulateMetadata(
    const at::Stack& stack,
    const OutputMetaDataVector& output_metadata) {
  m_output_metadata = output_metadata;
  if (UsesOutputMeta()) {
    const auto& meta = OutputMeta(stack);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(meta.size() == m_output_metadata.size());
    for (size_t i = 0; i < m_output_metadata.size(); ++i) {
      m_output_metadata[i].shape = meta[i].shape;
      m_output_metadata[i].dtype = meta[i].dtype;
      m_output_metadata[i].strides = meta[i].strides;
      m_output_metadata[i].mem_format = meta[i].mem_format;
      m_output_metadata[i].undefined = meta[i].undefined;
      m_output_metadata[i].exp_bias = meta[i].exp_bias;
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
    for (size_t i = 0; i < m_output_metadata.size(); ++i) {
      m_output_metadata[i].shape = outshapes[i];

      auto& dtype = m_output_metadata[i].dtype;
      if (c10::ScalarType::Undefined == dtype) {
        if (m_promote_type || m_promote_int_to_float) {
          dtype = habana_helpers::DTypeHelper::get_compute_dtype(
              stack,
              c10::nullopt,
              m_promote_int_to_float
                  ? habana_helpers::DTypeHelper::DtypePromoteVariant::
                        kPromoteIntToFloat
                  : habana_helpers::DTypeHelper::DtypePromoteVariant::
                        kPromoteToCommon,
              false,
              c10::nullopt,
              false,
              false);
        } else {
          // Use self's dtype if dtype is not propagated from frontend
          dtype = stack_tensor(stack, 0).scalar_type();
        }
      }

      HandleHwScaling(stack, i);
    }
  }
}

void OpBackend::AllocateAndAddSynapseNode(
    sh::graph& graph,
    at::Stack& stack,
    const OutputMetaDataVector& output_metadata) {
  PopulateMetadata(stack, output_metadata);

  HandleFn(graph);
  HandleInplaceFn(graph, stack);
  HandleOutFn(graph, stack);

  HandleScalarToTensor(graph, stack);

  if (!GET_ENV_FLAG_NEW(PT_DISABLE_DTYPE_PROMOTION)) {
    HandleTypePromotion(graph, stack);
  }

  CustomHandler(graph, stack);

  AddNode(graph, stack);
}

void OpBackend::CreateShapeTensorInput(
    sh::graph& graph,
    at::ScalarType dtype,
    at::IntArrayRef sizes,
    std::vector<synTensor>& inputs,
    synTensorType shape_tensor_type,
    bool force_create,
    void* hostDataPtr) {
  // Add intermediate shape tensor
  if (isOutputInfMode()) {
    auto& meta = GetOutputInfMeta();
    const auto& st = GetProxyTensor(dtype, sizes);
    const auto& md = TensorMetaData(
        st.sizes().vec(),
        st.strides().vec(),
        dtype,
        at::MemoryFormat::Contiguous);
    meta.AddShapeTensor(md);

    return;
  }

  if (force_create or graph.is_dynamic_graph()) {
    auto st = habana_helpers::create_shape_tensor(
        GetProxyTensor(dtype, sizes),
        graph,
        false,
        shape_tensor_type,
        std::string(),
        hostDataPtr);
    st.set_intermediate_shape_tensor();
    graph.increment_shape_tensors();
    m_shape_tensors.emplace_back(std::move(st));
    inputs.emplace_back(m_shape_tensors.back().get());
  }
}

void OpBackend::CreateH2dTensorInput(
    sh::graph& graph,
    at::ScalarType dtype,
    void* hostDataPtr,
    size_t hostDataSize,
    std::vector<synTensor>& inputs,
    synTensorType shape_tensor_type,
    bool force_create) {
  size_t es = c10::elementSize(dtype);
  int64_t tensorSize = (hostDataSize + es - 1) / es;

  CreateShapeTensorInput(
      graph,
      dtype,
      at::IntArrayRef(tensorSize),
      inputs,
      shape_tensor_type,
      force_create,
      hostDataPtr);
}

std::vector<sh::tensor> OpBackend::BuildNode(
    OpBackend* op,
    sh::graph& graph,
    NodeAttr&& node_attr) {
  op->m_num_syn_nodes++;
  if (op->isOutputInfMode()) {
    auto& meta = op->GetOutputInfMeta();
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

    // populate node params
    PT_BRIDGE_DEBUG(
        "OpBackend adding params data=",
        node_attr.params,
        ", params size=",
        node_attr.param_size);

    meta.AddNodeParams(node_attr.params, node_attr.param_size);

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

      const auto& t = GetProxyTensor(attr.dtype, attr.sizes, attr.exp_bias);
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
                        std::string(),
                        std::string())
                  : habana_helpers::create_tensor(
                        t,
                        graph,
                        is_persistent,
                        is_external,
                        attr.syn_data_type,
                        std::string(),
                        std::string()));

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

  graph.add_node(
      std::move(node_attr.inputs),
      std::move(node_outputs),
      node_attr.params,
      node_attr.param_size,
      node_attr.guid,
      nullptr,
      input_layouts.empty() ? nullptr : input_layouts.data(),
      output_layouts.empty() ? nullptr : output_layouts.data(),
      op->deterministic,
      op->getContextHints());

  return outputs;
}

sh::tensor OpBackend::BuildBoolCast(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    const at::IntArrayRef sizes,
    const at::ScalarType& from,
    c10::optional<int> final_result_index) {
  // We want either 0x00 or 0x01 stored in bytes when casting from or to Bool.
  auto zero_tensor = OpBackend::BuildConstant(op, graph, 0, from);

  auto eq = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("equal_fwd", from),
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

sh::tensor OpBackend::BuildRegularCast(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    const at::IntArrayRef sizes,
    const at::ScalarType& from,
    const at::ScalarType& to,
    c10::optional<int> final_result_index) {
  // Verify from and to types correctness
  BuildCastGuid(from, to);

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

    ns_CastKernel::ParamsV3 params{};
    params.round_mode = habana_helpers::get_cast_rounding_mode(to);
    auto device_type{habana::HPURegistrar::get_device().type()};
    if (sh::device_supports_trunc(device_type) &&
        src == c10::ScalarType::Float &&
        (dst == c10::ScalarType::Char || dst == c10::ScalarType::Byte)) {
      params.mode = CAST_TRUNC;
    }

    auto is_last = (i + 1) == cast_sequence.size();
    auto output_index = is_last ? final_result_index : c10::nullopt;
    NodeAttr castnode{
        cast_guid,
        {*input},
        {{sizes, dst, output_index}},
        &params,
        sizeof(params)};
    auto cast = BuildNode(op, graph, std::move(castnode));
    casts.emplace_back(std::move(cast.at(0)));
    input = &casts.back().get();
  }

  HABANA_ASSERT(!casts.empty(), "Empty vector of casts.");
  return std::move(casts.back());
};

sh::tensor OpBackend::BuildCast(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    const at::IntArrayRef sizes,
    const at::ScalarType& from,
    const at::ScalarType& to,
    c10::optional<int> final_result_index) {
  PT_BRIDGE_DEBUG("Performing cast from:\t", from, "\t\tto:\t", to);
  bool handle_from_bool =
      from == at::kBool && GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 0;
  if (!(handle_from_bool || to == at::kBool))
    return OpBackend::BuildRegularCast(
        op, graph, syn_in, sizes, from, to, final_result_index);

  auto boolResult = OpBackend::BuildBoolCast(
      op,
      graph,
      syn_in,
      sizes,
      from,
      to == at::kBool ? final_result_index : c10::nullopt);

  if (to == at::kBool || to == at::kChar)
    return boolResult;

  return OpBackend::BuildRegularCast(
      op, graph, boolResult.get(), sizes, at::kBool, to, final_result_index);
}

sh::tensor OpBackend::BuildConstant(
    OpBackend* op,
    sh::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    const at::IntArrayRef constant_outshape,
    c10::optional<int> final_result_index) {
  // For eager mode, Allocate constant synapse tensor
  // for non-persistent tensor of size {1}.
  if (op->GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER &&
      !final_result_index.has_value() && constant_outshape.equals({1})) {
    return OpBackend::BuildConstantTensor(op, graph, val, force_type);
  }

  at::ScalarType valtype =
      force_type.has_value() ? force_type.value() : val.type();

  std::vector<synTensor> input;
  op->CreateShapeTensorInput(graph, valtype, constant_outshape, input);

  if (valtype == c10::ScalarType::Long && common::IsInt64Supported()) {
    ns_ConstantKernel::Params_v2 paramsV2{};
    int64_t value = val.to<int64_t>();
    paramsV2.const_low = value;
    paramsV2.const_high = value >> 32;

    auto constant = BuildNode(
        op,
        graph,
        {get_guid_with_precision("constant", valtype, true),
         input,
         {{constant_outshape, valtype, final_result_index}},
         &paramsV2,
         sizeof(paramsV2)});
    return std::move(constant.at(0));
  } else {
    ns_ConstantKernel::Params params{};
    if (valtype == c10::ScalarType::Int or valtype == c10::ScalarType::Long) {
      get<int>(params.constant) = val.to<int>();
      if (habana_helpers::is_downcast_to_int_needed(valtype)) {
        valtype = c10::ScalarType::Int;
      }
    } else {
      get<float>(params.constant) = val.to<float>();
    }

    auto constant = BuildNode(
        op,
        graph,
        {get_guid_with_precision("constant", valtype),
         input,
         {{constant_outshape, valtype, final_result_index}},
         &params,
         sizeof(params)});

    return std::move(constant.at(0));
  }
}

sh::tensor OpBackend::BuildConstantTensor(
    OpBackend* op,
    sh::graph& graph,
    const at::Scalar& val,
    c10::optional<at::ScalarType> force_type,
    [[maybe_unused]] const at::IntArrayRef outshape) {
  if (op->isOutputInfMode()) {
    // dummy synapse tensor
    return sh::tensor::create_placeholder({1}, {1});
  }

  return op->AllocateConstantSynapseTensor(graph, val, force_type);
}

sh::tensor OpBackend::BuildBroadcast(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
  std::vector<synTensor> inputs = {syn_in};
  op->CreateShapeTensorInput(graph, dtype, sizes, inputs);
  NodeAttr::NodeOutputAttr out_attr{sizes, dtype, final_result_index};
  out_attr.exp_bias = exp_bias;

  auto broadcast = BuildNode(op, graph, {"broadcast", inputs, {out_attr}});
  return std::move(broadcast.at(0));
}

sh::tensor OpBackend::BuildPermute(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::IntArrayRef permutation,
    at::ScalarType dtype,
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
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
         final_result_index,
         DATA_TENSOR,
         syn_type_na,
         c10::nullopt,
         exp_bias}},
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
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
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
  NodeAttr::NodeOutputAttr out_attr{sizes, dtype, final_result_index};
  out_attr.exp_bias = exp_bias;

  auto reshape = BuildNode(op, graph, {"reshape", inputs, {out_attr}});
  return std::move(reshape.at(0));
}

sh::tensor OpBackend::BuildIdentity(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index,
    c10::optional<unsigned> exp_bias) {
  NodeAttr::NodeOutputAttr out_attr{sizes, dtype, final_result_index};
  out_attr.exp_bias = exp_bias;
  auto identity = BuildNode(op, graph, {"identity", {syn_in}, {out_attr}});
  return std::move(identity.at(0));
}

sh::tensor OpBackend::BuildSqueeze(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<unsigned> axis,
    c10::optional<int> final_result_index) {
  auto axisHasValue = axis.has_value();
  synAxisParams squeezeParams = {.axis = axisHasValue ? axis.value() : 0};
  auto squeeze = BuildNode(
      op,
      graph,
      {"squeeze",
       {syn_in},
       {{sizes, dtype, final_result_index}},
       axisHasValue ? &squeezeParams : nullptr,
       axisHasValue ? sizeof(squeezeParams) : 0});
  return std::move(squeeze.at(0));
}

sh::tensor OpBackend::BuildExpandDims(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    unsigned axis,
    c10::optional<int> final_result_index) {
  synAxisParams expandDimsParams = {.axis = axis};
  auto squeeze = BuildNode(
      op,
      graph,
      {"expand_dims",
       {syn_in},
       {{sizes, dtype, final_result_index}},
       &expandDimsParams,
       sizeof(expandDimsParams)});
  return std::move(squeeze.at(0));
}

sh::tensor OpBackend::BuildFlatten(
    OpBackend* op,
    sh::graph& graph,
    synTensor syn_in,
    at::IntArrayRef sizes,
    at::ScalarType dtype,
    c10::optional<int> final_result_index) {
  auto flatten = BuildNode(
      op,
      graph,
      {"flatten_fwd", {syn_in}, {{sizes, dtype, final_result_index}}});
  return std::move(flatten.at(0));
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

  std::string guid = get_guid_with_precision("non_zero_fwd", inScalarType);

  return op->BuildOp(
      graph,
      std::move(guid),
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
    c10::optional<int> finalResultIndex) {
  const auto& inputTensor = inTensors[0];
  const auto& indexTensor = inTensors[1];
  const auto& updatesTensor = inTensors[2];

  constexpr auto allowedNrOfInTensors = 3; // +1 optional

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

  std::string guid =
      get_guid_with_precision("scatter_nd_onnx_fwd", inScalarType);

  return std::move(op->BuildOp(
                         graph,
                         std::move(guid),
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
