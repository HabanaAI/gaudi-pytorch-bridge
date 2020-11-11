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
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/softmax_kernels.h"
#include "synapse_helpers/recipe.h"

#include <algorithm>

using namespace torch;

namespace habana {

void LogSoftmaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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

  auto output = habana_helpers::createPTTensor(self, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void LogSoftmaxBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
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

  // For logsoftmax_bwd_ kernel, the node inputs are in order {grad, output,
  // input} The synapse graph needs only the grad and output, and i the order
  // {output, grad} The p_context_->pt_inputs_ abd p_context_->syn_inputs_ are
  // modified here to ensure this.
  TORCH_CHECK(
      p_context_->pt_inputs_.size() == 3,
      "logsoftmax_bwd node should have 3 input pytorch tensors");
  TORCH_CHECK(
      p_context_->syn_inputs_.size() == 3,
      "logsoftmax_bwd node should have 3 input synapse tensors");
  // Remove the "input" tensor at the end
  p_context_->pt_inputs_.pop_back();
  p_context_->syn_inputs_.pop_back();

  // Reorder the grad and output
  std::swap(p_context_->pt_inputs_[0], p_context_->pt_inputs_[1]);
  std::swap(p_context_->syn_inputs_[0], p_context_->syn_inputs_[1]);
  auto grad_output =
      habana_helpers::createPTTensor(input, is_output_persistent);

  AllocateSynapseOutput(graph, grad_output, is_output_persistent);
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
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
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

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    std::vector<at::Tensor> pt_inputs{output, grad};
    auto output = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{grad, output, input};
    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

void SoftmaxOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Softmax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for Softmax operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be int for Softmax operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input type expected to be Bool for Softmax operator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  auto half_to_float = inputs[2].toBool();

  TORCH_CHECK(
      !half_to_float,
      "softmax with half to float conversion is not supported on HPU");
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
    CastOperator intToFloatOp(this->p_context_->device_id_, node_type);
    auto& float_syn =
        intToFloatOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

    // Build Params for the graph
    std::vector<c10::IValue> stack{IValue(self),
                                   IValue(c10::ScalarType::Float)};
    intToFloatOp.AllocateAndAddSynapseNode(graph, stack, false);

    synapse_helpers::tensor& float_syn_tensor = intToFloatOp.GetSynOutputs()[0];
    auto output_float = intToFloatOp.GetOutputs()[0];
    p_context_->syn_inputs_[0] = std::move(float_syn);

    auto output = at::empty(
        output_float.sizes(),
        output_float.options(),
        output_float.suggest_memory_format());
    AllocateSynapseOutput(graph, output, is_output_persistent);
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
        std::move(node_type));
  } else {
    // Softmax Operator
    p_context_->params_.emplace<ns_Softmax::Params>(params);
    p_context_->params_size_ = sizeof(params);

    auto output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, &params, sizeof(params));
  }
}

void SoftmaxOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Softmax operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for Softmax operator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input type expected to be int for Softmax operator");
  TORCH_CHECK(
      inputs[2].isBool(),
      "Input type expected to be Bool for Softmax operator");

  auto self = inputs[0].toTensor();
  auto output = at::empty(
      self.sizes(),
      self.options().dtype(c10::ScalarType::Float),
      self.suggest_memory_format());
  HabanaOperator::SetPTOutput(output);
}
/** softmax (forward pass) implementation for Habana device
 * @params [In] self: Input tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] half_to_float:
 */
Tensor softmax_hpu(const Tensor& self, int64_t dim, const bool half_to_float) {
  PT_KERNEL_BEGIN;

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "softmax_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  SoftmaxOperator Op(device_id, scalar_type);

  std::vector<c10::IValue> stack = {
      IValue(self), IValue(dim), IValue(half_to_float)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}

void SoftmaxBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of input expected for SoftmaxBackward operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input type expected to be tensor for SoftmaxBackward operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input type expected to be tensor for SoftmaxBackward operator");
  TORCH_CHECK(
      inputs[2].isInt(),
      "Input type expected to be int for SoftmaxBackward operator");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input type expected to be tensor for SoftmaxBackward operator");

  at::Tensor grad = inputs[0].toTensor();
  at::Tensor output = inputs[1].toTensor();
  int dim = inputs[2].toInt();
  at::Tensor input = inputs[3].toTensor();

  dim = at::maybe_wrap_dim(dim, input.dim(), /*wrap_scalar=*/true);

  ns_Softmax::Params params{static_cast<int>(input.ndimension() - 1 - dim)};

  p_context_->params_.emplace<ns_Softmax::Params>(params);
  p_context_->params_size_ = sizeof(params);

  // For softmax_bwd_ kernel, the node inputs are in order {grad, output,
  // input} The synapse graph needs only the grad and output, and in the order
  // {output, grad} The p_context_->pt_inputs_ and p_context_->syn_inputs_ need
  // to be modified to ensure this. Correct approch to do this is TBD.

  TORCH_CHECK(
      p_context_->pt_inputs_.size() == 3,
      "softmax_bwd node should have 3 input pytorch tensors");
  TORCH_CHECK(
      p_context_->syn_inputs_.size() == 3,
      "softmax_bwd node should have 3 input synapse tensors");
  // graph mode passes 3 inputs
  // Remove the "input" tensor at the end
  p_context_->pt_inputs_.pop_back();
  p_context_->syn_inputs_.pop_back();

  // Reorder the grad and output
  std::swap(p_context_->pt_inputs_[0], p_context_->pt_inputs_[1]);
  std::swap(p_context_->syn_inputs_[0], p_context_->syn_inputs_[1]);

  auto input_grad =
      at::empty(input.sizes(), input.options(), input.suggest_memory_format());

  AllocateSynapseOutput(graph, input_grad, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

/** softmax (backward pass) implementation for Habana device
 * @params [In] grad: Backward pass Input tensor. 2-4D. bf16, fp32
 * @params [In] output: Forward pass Output tensor. 2-4D. bf16, fp32
 * @params [In] dim: Dimension along which softmax will be computed
 * @params [In] input: Forward pass Input tensor. 2-4D. bf16, fp32
 */
Tensor softmax_backward_hpu(
    const Tensor& grad,
    const Tensor& output,
    int64_t dim,
    const Tensor& input) {
  PT_KERNEL_BEGIN;

  size_t device_id = grad.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  at::ScalarType scalar_type = grad.scalar_type();
  std::string node_type =
      "softmax_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // create the operator
  SoftmaxBackwardOperator Op(device_id, scalar_type);
  std::vector<c10::IValue> stack = {
      IValue(grad), IValue(output), IValue(dim), IValue(input)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    // For softmax_bwd_ kernel, the node inputs are in order {grad, output,
    // input} The synapse graph needs only the grad and output, and in the order
    // {output, grad}. Correct way to set the Inputs is TBD
    std::vector<at::Tensor> pt_inputs{output, grad};
    auto input_grad = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(input_grad);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // create graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    // Assign Inputs to the Operator
    std::vector<at::Tensor> pt_inputs{grad, output, input};
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  PT_KERNEL_END;

  return out.at(0);
}
} // end namespace habana

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "aten::log_softmax",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::LogSoftmaxOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::_log_softmax_backward_data",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::LogSoftmaxBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::_softmax",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::SoftmaxOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::_softmax_backward_data",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<habana::SoftmaxBackwardOperator>(
                  device_id, node_type);
            });
