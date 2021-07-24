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
namespace habana {

template <typename T>
T& get(fint_t&);

template <>
int& get<int>(fint_t& u) {
  return u.i;
}
template <>
float& get<float>(fint_t& u) {
  return u.f;
}

#define PARAMS_STUB(structname) \
  size = sizeof(structname);    \
  auto params = std::make_shared<structname>()

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
      "constant_" + habana_helpers::name_suffix_from_type(m_scalar_type),
      graph,
      {},
      {{1, m_scalar_type, false}},
      params.get(),
      size);

  // Set output from constant as input to this node at index m_scalar_id
  p_context_->syn_inputs_.emplace(
      p_context_->syn_inputs_.cbegin() + m_scalar_id, std::move(const_out[0]));
}

void HabanaOperatorHelper::HandleFn(
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    bool is_output_persistent) {
  if (m_out_id < 0) {
    return;
  }

  const auto& output = habana_helpers::createPTTensor(
      stack.at(m_out_id).toTensor(), is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
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
    bool) {
  size_t size = 0;
  const auto& params = FillParams(stack, size);
  AddNodeToSynapseGraph(graph, params.get(), size);
}

void HabanaOperatorHelper::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    bool is_output_persistent) {
  CustomHandler(graph, stack);
  HandleFn(graph, stack, is_output_persistent);
  HandleInplaceFn(stack);
  HandleOutFn(stack);
  HandleScalarToTensor(graph, stack);

  AddNode(graph, stack, is_output_persistent);
}

std::vector<synapse_helpers::tensor> HabanaOperatorHelper::BuildOp(
    std::string guid,
    synapse_helpers::graph& graph,
    std::vector<synTensor> syn_in,
    const std::vector<_intermediate_attr>& out_props,
    void* params,
    size_t param_size) {
  std::vector<synapse_helpers::tensor> outputs;
  std::vector<synTensor> syn_out;

  for (const auto& out_prop : out_props) {
    const auto& t = at::detail::make_tensor<c10::TensorImpl>(
        c10::DispatchKeySet{
            at::DispatchKey::HPU, at::DispatchKey::AutogradHABANA},
        c10::scalarTypeToTypeMeta(out_prop.dtype),
        c10::Device(c10::kHABANA, 0));
    t.unsafeGetTensorImpl()->set_sizes_contiguous(out_prop.sizes);
    outputs.emplace_back(habana_helpers::create_tensor(
        t, graph.get_graph_handle(), out_prop.persistent, out_prop.dtype));
    syn_out.emplace_back(outputs.back().get());
  }

  auto result = graph.add_node(
      std::move(syn_in), std::move(syn_out), params, param_size, guid);
  HABANA_ASSERT(
      ok(result),
      "Adding ",
      guid,
      " to graph failed with ",
      get_error(result).error);

  return outputs;
}
template <typename ScalarType>
static std::shared_ptr<void> ClampParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_ClampKernel::Params);

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampParams(
    const at::Stack& stack,
    size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    float min = stack[1].isScalar() ? stack[1].toScalar().to<float>()
                                    : -std::numeric_limits<float>::max();
    float max = stack[2].isScalar() ? stack[2].toScalar().to<float>()
                                    : std::numeric_limits<float>::max();
    return ClampParams(min, max, size);
  } else {
    int min = stack[1].isScalar() ? stack[1].toScalar().to<int>()
                                  : -std::numeric_limits<int>::max();
    int max = stack[2].isScalar() ? stack[2].toScalar().to<int>()
                                  : std::numeric_limits<int>::max();
    return ClampParams(min, max, size);
  }
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampMinParams(
    const at::Stack& stack,
    size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
  }
  return ClampParams(
      stack[1].toScalar().toInt(), std::numeric_limits<int>::max(), size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillClampMaxParams(
    const at::Stack& stack,
    size_t& size) {
  if (c10::isFloatingType(stack[0].toTensor().scalar_type())) {
    return ClampParams(
        -std::numeric_limits<float>::max(),
        stack[1].toScalar().toFloat(),
        size);
  }
  return ClampParams(
      -std::numeric_limits<int>::max(), stack[1].toScalar().toInt(), size);
}

std::shared_ptr<void> HabanaOperatorHelper::FillHardSigmoidParams(
    const at::Stack&,
    size_t& size) {
  PARAMS_STUB(ns_HardSigmoidKernel::Params);
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;

  params->alpha = alpha;
  params->beta = beta;

  return params;
}

std::shared_ptr<void> HabanaOperatorHelper::FillMseLossParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MSELossKernel::Params);

  auto mode = stack.at(stack.at(2).isInt() ? 2 : 3).toInt();
  switch (mode) {
    case at::Reduction::Reduction::None:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_NONE;
      break;
    case at::Reduction::Reduction::Mean:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_MEAN;
      break;
    case at::Reduction::Reduction::Sum:
      params->mode = MSELossMode_t::MSE_LOSS_REDUCTION_MODE_SUM;
      break;
    default:
      TORCH_CHECK(false, "Unsupported reduction mode in mseloss: ", mode);
  }
  return params;
}

sizes_vec HabanaOperatorHelper::MseLossOutputShape(
    const torch::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return {self.sizes().vec()};
  }
  return {{}};
}

sizes_vec HabanaOperatorHelper::PowOutputShape(const torch::Tensor& self) {
  return {self.sizes().vec()};
}

std::shared_ptr<void> HabanaOperatorHelper::FillCumsumParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_CumSumKernel::Params);
  auto self = stack.at(0).toTensor();
  auto dim = at::maybe_wrap_dim(stack.at(1).toInt(), self.dim(), true);
  params->axis = static_cast<int>(self.sizes().vec().size() - dim - 1);

  return params;
}

void BinaryWithAlphaOutOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    bool is_output_persistent) {
  if (ScalarInputs().at(ScalarId()).toFloat() == 1.) {
    p_context_->syn_inputs_.erase(
        p_context_->syn_inputs_.cbegin() + ScalarId());
    return HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent);
  }

  auto mul = BuildOp(
      "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      graph,
      {syn_in(1), syn_in(2)},
      {{stack_tensor(stack, 1).sizes(), ScalarType(), false}});

  auto op = BuildOp(
      guid_,
      graph,
      {syn_in(0), mul[0].get()},
      {{stack_tensor(stack, 0).sizes(), ScalarType(), is_output_persistent}});

  syn_out(0) = std::move(op[0]);
}

void RsubOp::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    bool is_output_persistent) {
  if (ScalarInputs().at(ScalarId()).toFloat() == 1.) {
    p_context_->syn_inputs_.erase(
        p_context_->syn_inputs_.cbegin() + ScalarId());
    std::swap(syn_in(0), syn_in(1));
    return HabanaOperatorHelper::AddNode(graph, stack, is_output_persistent);
  }
  auto mul = BuildOp(
      "mult_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      graph,
      {syn_in(0), syn_in(2)},
      {{stack_tensor(stack, 1).sizes(), ScalarType(), false}});

  auto op = BuildOp(
      guid_,
      graph,
      {syn_in(1), mul[0].get()},
      {{stack_tensor(stack, 0).sizes(), ScalarType(), is_output_persistent}});

  syn_out(0) = std::move(op[0]);
}
} // namespace habana
