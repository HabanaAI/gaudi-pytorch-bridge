/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/softmax_kernels.h"
#include "synapse_helpers/recipe.h"

#include <algorithm>

using namespace torch;
using namespace habana;

std::vector<int64_t> LogSoftmaxOperator::compute_output_shape(
    const Tensor& self) {
  return self.sizes().vec();
}

void LogSoftmaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  bool half_to_float;
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for softmax operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isInt(), "Input type expected to be int");
  if (!inputs[2].isNone())
    TORCH_CHECK(inputs[2].isBool(), "Input type expected to be Bool");

  at::Tensor self = inputs[0].toTensor();
  int dim = inputs[1].toInt();

  // FIXME Need to fix it for GraphMode SW-13887

  if (!inputs[2].isNone()) {
    half_to_float = inputs[2].toBool();
    TORCH_CHECK(
        !half_to_float,
        "softmax with half to float conversion is not supported on HPU");
  }

  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  p_context_->params_.emplace<ns_Softmax::Params>(params);
  p_context_->params_size_ = sizeof(params);

  auto output =
      habana_helpers::createPTTensor(self, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::vector<int64_t> LogSoftmaxBackwardOperator::compute_output_shape(
    const Tensor& input) {
  return input.sizes().vec();
}

void LogSoftmaxBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for softmax operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isInt(), "Input type expected to be int");
  TORCH_CHECK(inputs[3].isTensor(), "Input type expected to be tensor");

  at::Tensor grad = inputs[0].toTensor();
  at::Tensor output = inputs[1].toTensor();
  int dim = inputs[2].toInt();
  at::Tensor input = inputs[3].toTensor();

  dim = at::maybe_wrap_dim(dim, input.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(input.ndimension() - 1 - dim)};

  p_context_->params_.emplace<ns_Softmax::Params>(params);
  p_context_->params_size_ = sizeof(params);

  auto grad_output =
      habana_helpers::createPTTensor(input, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, grad_output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/** log_softmax (forward pass) implementation for Habana device
 * @params [In] self: Input tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] half_to_float:
 */
Tensor log_softmax_hpu(
    const Tensor& self,
    const int64_t dim,
    const bool half_to_float) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "logsoftmax_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // create the operator
  LogSoftmaxOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(half_to_float)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  std::vector<at::Tensor> pt_inputs{self};
  if (device.get_recipe_handle_cache().isCached(key)) {
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.Execute(key, pt_inputs, output);
  } else {
    // Build Params for the graph
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

/** log_softmax (backward pass) implementation for Habana device
 * @params [In] grad: Backward pass Input tensor. 2-4D. bf16, fp32
 * @params [In] output: Forward pass Output tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] input: Forward pass Input tensor. 2-4D. bf16, fp32
 */
Tensor log_softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_KERNEL_BEGIN;

  size_t device_id = grad.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = grad.scalar_type();
  std::string node_type =
      "logsoftmax_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  // create the operator
  LogSoftmaxBackwardOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(grad), IValue(output), IValue(dim), IValue(input)};
  size_t key = Op.GetRecipeKey(node_type, stack);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad, output, input};

  if (device.get_recipe_handle_cache().isCached(key)) {
    auto output = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op.Execute(key, pt_inputs, output);
  } else {
    // create graph
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

void SoftmaxIntOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Softmax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for Softmax operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be int for Softmax operator");
  TORCH_CHECK(
      inputs[2].isScalar() || inputs[2].isInt() || inputs[2].isNone(),
      "Input type expected to be Scalar for SoftmaxInt operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  if (!inputs[2].isNone()) {
    if (inputs[2].isScalar()) {
      [[maybe_unused]] auto dtype = inputs[2].toScalar();
    } else if (inputs[2].isInt()) {
      [[maybe_unused]] auto dtype = inputs[2].toInt();
    }
  }
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(self.ndimension() - 1 - dim)};

  if (self.scalar_type() == c10::ScalarType::Int ||
      self.scalar_type() == c10::ScalarType::Bool) {
    // Cast Input tensor to Float tensor
    std::string node_type;
    if (self.scalar_type() == c10::ScalarType::Int) {
      node_type = "cast_i32_to_f32";
    } else if (self.scalar_type() == c10::ScalarType::Bool) {
      node_type = "cast_i8_to_f32";
    }

    // Create the operator
    auto intToFloatOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    intToFloatOp->SetSynapseInput(p_context_->syn_inputs_[0]);

    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::ScalarType::Float)};
    intToFloatOp->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));

    synapse_helpers::tensor& float_syn_tensor =
        intToFloatOp->GetSynOutputs()[0];
    auto output_float = intToFloatOp->GetOutputs()[0];

    auto output = habana_helpers::createPTTensor(
        output_float, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    at::ScalarType scalar_type = output_float.scalar_type();
    node_type =
        "softmax_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &params,
        sizeof(params),
        std::move(node_type),
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  } else {
    // Softmax Operator
    p_context_->params_.emplace<ns_Softmax::Params>(params);
    p_context_->params_size_ = sizeof(params);

    auto output =
        habana_helpers::createPTTensor(self, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void SoftmaxIntOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Softmax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for Softmax operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be int for Softmax operator");
  TORCH_CHECK(
      inputs[2].isScalar(),
      "Input type expected to be Bool for SoftmaxInt operator");

  auto self = inputs[0].toTensor();
  if (self.dtype() == c10::ScalarType::BFloat16) {
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::BFloat16),
        self.suggest_memory_format());
    HabanaOperator::SetPTOutput(output);
  } else {
    auto output = at::empty(
        self.sizes(),
        self.options().dtype(c10::ScalarType::Float),
        self.suggest_memory_format());
    HabanaOperator::SetPTOutput(output);
  }
}

Tensor softmax_int_hpu(
    const Tensor& self,
    int64_t dim,
    c10::optional<ScalarType> dtype) {
  PT_KERNEL_BEGIN;
  static_cast<void>(dtype);
  auto result = torch::_softmax(self, dim, false);
  PT_KERNEL_END;
  return result;
}

static auto& SoftmaxKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::log_softmax", KERNEL_FN(LogSoftmaxOperator))
        .add("aten::softmax", KERNEL_FN(SoftmaxIntOperator))
        .add("aten::softmax.int", KERNEL_FN(SoftmaxIntOperator));
