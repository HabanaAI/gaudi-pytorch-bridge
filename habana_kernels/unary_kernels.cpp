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

#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_composite_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_kernels/unary_kernels.h"
#include "synapse_helpers/recipe.h"

using namespace torch;
using namespace torch::jit;

void UnaryOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inpust expected for Relu operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  at::Tensor input = inputs[0].toTensor();

  auto output = habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void UnaryInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  AllocateSynapseInplaceOutput(graph);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

Tensor unary_op_hpu(
    const Tensor& input,
    std::string& node_type,
    UnaryOperator* Op) {
  size_t device_id = input.device().index();
  std::vector<c10::IValue> stack = {IValue(input)};
  std::vector<at::Tensor> pt_inputs{input};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op->GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutput(output);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    //
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
  std::vector<at::Tensor> out = Op->GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  return out.at(0);
}

void unary_inplace_op_hpu(
    const Tensor& self,
    std::string& node_type,
    UnaryInplaceOperator* Op) {
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};

  size_t key = Op->GetRecipeKey(node_type, stack, true);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutput(pt_inputs[0]);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
}

void UnaryBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inpust expected for UnaryBackward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");

  at::Tensor grad_in = inputs[0].toTensor();
  at::Tensor input = inputs[1].toTensor();

  TORCH_CHECK(
      grad_in.scalar_type() == input.scalar_type(),
      "Types don't match. grad_in type: ",
      grad_in.scalar_type(),
      " input type: ",
      input.scalar_type());
  TORCH_CHECK(
      (grad_in.sizes() == input.sizes()) ||
          (grad_in.ndimension() == input.ndimension() &&
           std::all_of(
               input.sizes().cbegin(),
               input.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. grad_in sizes: ",
      grad_in.sizes(),
      ", input sizes: ",
      grad_in.sizes());

  auto grad_output =
      habana_helpers::createPTTensor(input, is_output_persistent);
  AllocateSynapseOutput(graph, grad_output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

Tensor unary_backward_op_hpu(
    const Tensor& grad_in,
    const Tensor& input,
    std::string& node_type,
    UnaryBackwardOperator* Op) {
  TORCH_CHECK(
      grad_in.scalar_type() == input.scalar_type(),
      "Types don't match. grad_in type: ",
      grad_in.scalar_type(),
      " input type: ",
      input.scalar_type());
  TORCH_CHECK(
      (grad_in.sizes() == input.sizes()) ||
          (grad_in.ndimension() == input.ndimension() &&
           std::all_of(
               input.sizes().cbegin(),
               input.sizes().cend(),
               [](auto val) { return val == 1; })),
      "Sizes in elementwise kernel don't match. grad_in sizes: ",
      grad_in.sizes(),
      ", input sizes: ",
      input.sizes());
  size_t device_id = input.device().index();
  std::vector<c10::IValue> stack = {IValue(grad_in), IValue(input)};
  std::vector<at::Tensor> pt_inputs{grad_in, input};
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  size_t key = Op->GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = at::empty(
        input.sizes(), input.options(), input.suggest_memory_format());
    Op->SetPTInputs(pt_inputs);
    Op->SetPTOutput(output);
    Op->Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    //
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    // Assign Inputs to the Operator
    Op->AllocateSynapseInputs(graph, pt_inputs, true);

    // Build Params for the graph
    Op->AllocateAndAddSynapseNode(graph, stack, true);

    // compile and execute the graph
    Op->Compile(graph);
  }
  std::vector<at::Tensor> out = Op->GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  return out.at(0);
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.relu(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor relu_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "relu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  ReluOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.relu_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& relu_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "relu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  ReluInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);
  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sigmoid(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sigmoid_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "sigmoid_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  SigmoidOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sigmoid(grad_in, input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] grad_in - input tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sigmoid_backward_hpu(const Tensor& grad_in, const Tensor& input) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "sigmoid_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  SigmoidBackwardOperator Op(device_id, scalar_type);

  auto out = unary_backward_op_hpu(grad_in, input, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.sqrt(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor sqrt_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "sqrt_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  SqrtOperator Op(device_id, scalar_type);

  auto output = unary_op_hpu(input, node_type, &Op);

  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.tanh(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor tanh_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "tanh_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  TanhOperator Op(device_id, scalar_type);

  auto output = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return output;
}

/*************************************************************************
 * @brief Kernel implementation for output = a.tanh(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/

Tensor& tanh_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for torch.tanh(input,out)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/

Tensor& tanh_out_hpu(Tensor& out, Tensor& self) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self};
  std::vector<at::Tensor> pt_outputs{out};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "tanh", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.tanh(grad_in, input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] grad_in - input tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor tanh_backward_hpu(const Tensor& grad_in, const Tensor& input) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "tanh_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  TanhBackwardOperator Op(device_id, scalar_type);

  auto grad_output = unary_backward_op_hpu(grad_in, input, node_type, &Op);

  PT_KERNEL_END;
  return grad_output;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.floor(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor floor_hpu(const Tensor& input) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = input.scalar_type();
  std::string node_type =
      "floor_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = input.device().index();
  FloorOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(input, node_type, &Op);
  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.floor_(input)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& floor_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "floor_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();

  // Create the operator
  FloorInplaceOperator Op(device_id, scalar_type);
  unary_inplace_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return self;
}

void GeluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Gelu operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu operator");

  auto self = inputs[0].toTensor();

  auto output1 = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);

  // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
  // In graph mode we want 2nd output to be non-persistent to reduce memory
  // consumption
  auto output2 = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      isEagerMode() ? is_output_persistent : false);

  std::vector<at::Tensor> outputs{output1, output2};
  AllocateSynapseOutputs(graph, outputs, {is_output_persistent, false});
  AddNodeToSynapseGraph(graph, nullptr, 0);
  p_context_->excluded_output_indices_.insert(1);
}

/*************************************************************************
 * @brief Kernel implementation for gelu
 *output = 0.5 * x *(1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor gelu_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gelu_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  GeluOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
    auto output1 =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    auto output2 =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());

    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs({output1, output2});
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    Op.SetEagerMode();
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

void GeluBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Gelu Backward operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu Backward operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for Gelu Backward operator");

  auto grad = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  at::ScalarType scalar_type = self.scalar_type();

  // x^3 implemented as x*x*x. Identity node used to create aliased tensor
  // since GC/TPC does not like giving same tensor as both inputs to a
  // binary op
  IdentityOperator identityOp(this->p_context_->device_id_, scalar_type);
  auto& syn_arg0 =
      identityOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  torch::jit::Stack stack = {IValue(self)};
  identityOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(syn_arg0);
  stack.clear();

  MulOperator mulpow1Op(this->p_context_->device_id_, scalar_type);
  auto& mul_syn_1 =
      mulpow1Op.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  UNUSED auto& mul_syn_2 =
      mulpow1Op.SetSynapseInput(std::move(identityOp.GetSynOutputs()[0]));
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(identityOp.GetOutputs()[0]));
  mulpow1Op.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(mul_syn_1);
  stack.clear();

  MulOperator mulpow2Op(this->p_context_->device_id_, scalar_type);
  auto& mul_syn_11 =
      mulpow2Op.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  UNUSED auto& mul_syn_21 =
      mulpow2Op.SetSynapseInput(std::move(mulpow1Op.GetSynOutputs()[0]));
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(mulpow1Op.GetOutputs()[0]));
  mulpow2Op.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(mul_syn_11);
  stack.clear();

  // Create Add operator
  AddOperator addOp(this->p_context_->device_id_, scalar_type);
  auto& add_syn = addOp.SetSynapseInput(std::move(p_context_->syn_inputs_[1]));
  addOp.SetSynapseInput(std::move(mulpow2Op.GetSynOutputs()[0]));
  // Build Params for the graph
  Scalar alphaValue = 0.044715;
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(mulpow2Op.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue));
  addOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[1] = std::move(add_syn);
  stack.clear();

  // Create Mul operator
  MulOperator mulOp(this->p_context_->device_id_, scalar_type);
  mulOp.SetSynapseInput(std::move(addOp.GetSynOutputs()[0]));
  // Build Params for the graph
  Scalar alphaValue_2 = M_2_SQRTPI * M_SQRT1_2;
  stack.emplace_back(IValue(addOp.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue_2));
  mulOp.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // Create Tanh operator
  TanhOperator tanhOp(this->p_context_->device_id_, scalar_type);
  tanhOp.SetSynapseInput(std::move(mulOp.GetSynOutputs()[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  tanhOp.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  auto output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      is_output_persistent);
  AllocateSynapseOutput(graph, output, is_output_persistent);
  synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

  synapse_helpers::tensor& synInput0 = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& synInput1 = p_context_->syn_inputs_[1];
  synapse_helpers::tensor& synInput2 = tanhOp.GetSynOutputs()[0];

  std::vector<synTensor> syn_in{
      synInput0.get(), synInput1.get(), synInput2.get()};
  std::vector<synTensor> syn_out{synOutput.get()};

  graph.add_node(
      std::move(syn_in), std::move(syn_out), nullptr, 0, std::move(guid_));
}

/*************************************************************************
 * @brief Kernel implementation for gelu_backward
 * @param [out] output - bwd output tensor, 1-4D, BF16/FP32
 * @param [in] grad - bwd input tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor gelu_backward_hpu(const Tensor& grad, const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "gelu_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  GeluBackwardOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(grad), IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{grad, self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
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

/*************************************************************************
 * @brief Kernel implementation for erf_
 * output = x.erf_()
 * erf(x) = tanh((2/sqrt(pi))*(x+0.08943*x^3))
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& erf_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  Tensor self_copy =
      at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  habana_helpers::copy_data_within_device(self, self_copy, true);

  self.pow_(3.0).mul_(0.08943).add_(self_copy).mul_(M_2_SQRTPI).tanh_();

  PT_KERNEL_END;
  return self;
}

void ErfOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1, "Incorrect size of inputs expected for Erf operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Erf operator");

  auto self = inputs[0].toTensor();
  at::ScalarType scalar_type = self.scalar_type();

  // Create Pow operator
  PowOperator powOp(this->p_context_->device_id_, scalar_type);
  auto& pow_syn = powOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  // Build Params for the graph
  Scalar powValue = 3.0;
  std::vector<c10::IValue> stack{IValue(self), IValue(powValue)};
  powOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[0] = std::move(pow_syn);
  stack.clear();

  // Create Mul operator
  MulOperator mulOp(this->p_context_->device_id_, scalar_type);
  mulOp.SetSynapseInput(std::move(powOp.GetSynOutputs()[0]));
  // Build Params for the graph
  Scalar alphaValue = 0.08943;
  stack.emplace_back(IValue(powOp.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue));
  mulOp.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // Create Add operator
  AddOperator addOp(this->p_context_->device_id_, scalar_type);
  auto& add_syn = addOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));
  addOp.SetSynapseInput(std::move(mulOp.GetSynOutputs()[0]));
  // Build Params for the graph
  Scalar alphaValue_2 = 1.0;
  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue_2));
  addOp.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[0] = std::move(add_syn);
  stack.clear();

  // Create Mul operator
  MulOperator mulOp2(this->p_context_->device_id_, scalar_type);
  mulOp2.SetSynapseInput(std::move(addOp.GetSynOutputs()[0]));
  // Build Params for the graph
  Scalar alphaValue_3 = M_2_SQRTPI;
  stack.emplace_back(IValue(addOp.GetOutputs()[0]));
  stack.emplace_back(IValue(alphaValue_3));
  mulOp2.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // Create Tanh operator
  TanhOperator tanhOp(this->p_context_->device_id_, scalar_type);
  tanhOp.SetSynapseInput(std::move(mulOp2.GetSynOutputs()[0]));
  // Build Params for the graph
  stack.emplace_back(IValue(mulOp2.GetOutputs()[0]));
  tanhOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

  p_context_->syn_outputs_.emplace_back(std::move(tanhOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(tanhOp.GetOutputs()[0]));
}

Tensor erf_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "erf_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  ErfOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
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

/*************************************************************************
 * @brief Kernel implementation for exp_
 * output = x.exp_()
 * @param [out, in]  1-4D, BF16/FP32
 ************************************************************************/
Tensor& exp_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;

  auto self_copy =
      at::empty(self.sizes(), self.options(), self.suggest_memory_format());
  habana_helpers::copy_data_within_device(self, self_copy, true);

  std::vector<at::Tensor> pt_outputs{self};
  std::vector<at::Tensor> pt_inputs{self_copy};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "exp", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

void ExpOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1, "Incorrect size of inputs expected for Exp operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Exp operator");

  auto self = inputs[0].toTensor();
  auto output = habana_helpers::createPTTensor(self, is_output_persistent);

  AllocateSynapseOutput(graph, output, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

Tensor exp_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "exp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  ExpOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
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

/*************************************************************************
 * @brief Kernel implementation for torch.neg(input,out)
 * @param [out] out - output tensor, 1-4D, BF16/FP32
 * @param [in] input - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& neg_out_hpu(Tensor& result, const Tensor& input) {
  PT_KERNEL_BEGIN;

  // Resize result to correct size (if required)
  auto shape = DimVector(input.sizes());
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  std::vector<at::Tensor> pt_outputs{result};
  std::vector<at::Tensor> pt_inputs{input};

  synapse_simple_generic_kernel(
      pt_outputs, pt_inputs, "neg", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return result;
}

/*************************************************************************
 * @brief Kernel implementation for inplace torch.reciprocal_(self)
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& reciprocal_hpu_(Tensor& self) {
  PT_KERNEL_BEGIN;
  std::vector<at::Tensor> pt_inputs{self};

  synapse_simple_generic_inplace_kernel(
      pt_inputs, "reciprocal", nullptr, 0, SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;
  return self;
}

void ReciprocalOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Reciprocal operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Reciprocal operator");

  auto self = inputs[0].toTensor();
  auto result = habana_helpers::createPTTensor(self, is_output_persistent);
  inputs.insert(inputs.begin(), IValue(result));

  ReciprocalOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, is_output_persistent);
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.reciprocal(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor reciprocal_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  // create the operator
  ReciprocalOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto result =
        at::empty(self.sizes(), self.options(), self.suggest_memory_format());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
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

void ReciprocalOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for ReciprocalOut operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for ReciprocalOut operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for ReciprocalOut operator");

  auto result = inputs[0].toTensor();
  auto self = inputs[1].toTensor();

  auto shape = DimVector(self.sizes());
  auto tht_result = result.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  AllocateSynapseOutput(graph, result, is_output_persistent);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

/*************************************************************************
 * @brief Kernel implementation for torch.reciprocal(self,out)
 * @param [out] out - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor& reciprocal_out_hpu(Tensor& result, const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "reciprocal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  ReciprocalOutOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(result), IValue(self)};
  size_t key = Op.GetRecipeKey(node_type, stack);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(result);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
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

void ClampOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for Clamp operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  auto min = inputs[1].isScalar() ? inputs[1].toScalar()
                                  : inputs[1].toOptional<Scalar>();
  auto max = inputs[2].isScalar() ? inputs[2].toScalar()
                                  : inputs[2].toOptional<Scalar>();

  ns_ClampKernel::Params param;
  param.upperBound.f = max.value().to<float>();
  param.lowerBound.f = min.value().to<float>();

  if (self.scalar_type() == c10::ScalarType::Int) {
    // Guid needs to be updated since TPC only supports F32/BF16
    SetGuid("clamp_fwd_f32");
    // Cast Input tensor to Float tensor
    std::string node_type = "cast_i32_to_f32";

    // Create the operator
    CastOperator intToFloatOp(this->p_context_->device_id_, node_type);
    auto& float_syn =
        intToFloatOp.SetSynapseInput(std::move(p_context_->syn_inputs_[0]));

    // Build Params for the graph
    std::vector<c10::IValue> stack{
        IValue(self), IValue(c10::ScalarType::Float)};
    intToFloatOp.AllocateAndAddSynapseNode(graph, stack, false);

    synapse_helpers::tensor& float_syn_tensor = intToFloatOp.GetSynOutputs()[0];
    auto output_float = intToFloatOp.GetOutputs()[0];
    p_context_->syn_inputs_[0] = std::move(float_syn);
    stack.clear();

    AllocateSynapseOutput(graph, output_float, false);
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    node_type = "clamp_fwd_f32";
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        std::move(node_type));

    node_type = "cast_f32_to_i32";
    // Create cast operator
    CastOperator floatToIntOp(this->p_context_->device_id_, node_type);
    floatToIntOp.SetSynapseInput(std::move(p_context_->syn_outputs_[0]));

    // Build Params for the graph
    stack = {IValue(p_context_->pt_outputs_[0]), IValue(c10::ScalarType::Int)};

    floatToIntOp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_outputs_[0] = std::move(floatToIntOp.GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(floatToIntOp.GetOutputs()[0]);

  } else {
    auto output = habana_helpers::createPTTensor(self, is_output_persistent);
    AllocateSynapseOutput(graph, output, is_output_persistent);
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  }
}
/** @brief This function implements torch.clamp_min()
 * @param self (bf16, fp32 tensor) Input tensor
 * @param min (int, float) Minimum value at which input will be clamped
 */
Tensor clamp_min_hpu(const Tensor& self, Scalar min) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  ClampOperator Op(device_id, scalar_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self),
      IValue(min),
      IValue(Scalar(std::numeric_limits<float>::max()))};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
}

Tensor clamp_hpu(
    const Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);
  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  ClampOperator Op(device_id, scalar_type);

  std::vector<at::Tensor> pt_inputs;
  std::vector<c10::IValue> stack;
  if (self.scalar_type() == ScalarType::Long) {
    auto self_i32 = habana_helpers::cast_tensor_to_integer(self);
    pt_inputs = {self_i32};
    stack = {IValue(self_i32), IValue(min), IValue(max)};
  } else {
    pt_inputs = {self};
    stack = {IValue(self), IValue(min), IValue(max)};
  }

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);

    if (self.scalar_type() == ScalarType::Long) {
      auto output = at::empty(
          self.sizes(),
          self.options().dtype(c10::ScalarType::Int),
          self.suggest_memory_format());
      Op.SetPTOutput(output);
    } else {
      auto output =
          at::empty(self.sizes(), self.options(), self.suggest_memory_format());
      Op.SetPTOutput(output);
    }
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  if (self.scalar_type() == ScalarType::Long) {
    auto output = habana_helpers::cast_tensor_to_long(out.at(0));
    PT_KERNEL_END;
    return output;
  } else {
    PT_KERNEL_END;
    return out.at(0);
  }
}

void ClampInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for Clamp operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto input = inputs[0].toTensor();
  auto min = inputs[1].isScalar() ? inputs[1].toScalar()
                                  : inputs[1].toOptional<Scalar>();
  auto max = inputs[2].isScalar() ? inputs[2].toScalar()
                                  : inputs[2].toOptional<Scalar>();

  ns_ClampKernel::Params param;
  param.upperBound.f = max.value().to<float>();
  param.lowerBound.f = min.value().to<float>();

  AllocateSynapseInplaceOutput(graph);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

Tensor& clamp_hpu_(
    Tensor& self,
    c10::optional<Scalar> min,
    c10::optional<Scalar> max) {
  PT_KERNEL_BEGIN;

  if ((self.scalar_type() == ScalarType::Long) ||
      (self.scalar_type() == ScalarType::Int)) {
    self.copy_(clamp_hpu(self, min, max));
    PT_KERNEL_END;
    return self;
  }
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "clamp_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  ClampInplaceOperator Op(device_id, scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(min), IValue(max)};

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(self);
    Op.Execute(key);
  } else {
    PT_KERNEL_DEBUG("Key:", key);
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return self;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.abs(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor abs_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "abs_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  AbsOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

/*************************************************************************
 * @brief Kernel implementation for output = torch.neg(self)
 * @param [out] output - output tensor, 1-4D, BF16/FP32
 * @param [in] self - input tensor, 1-4D, BF16/FP32
 ************************************************************************/
Tensor neg_hpu(const Tensor& self) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "neg_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  // Create the operator
  size_t device_id = self.device().index();
  NegOperator Op(device_id, scalar_type);

  auto out = unary_op_hpu(self, node_type, &Op);

  PT_KERNEL_END;
  return out;
}

static auto& KernelRegistry =
    ::habana::KernelRegistry()
        .add(
            "aten::relu",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ReluOperator>(device_id, node_type);
            })
        .add(
            "aten::sigmoid",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SigmoidOperator>(device_id, node_type);
            })
        .add(
            "aten::sigmoid_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SigmoidBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::abs",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<AbsOperator>(device_id, node_type);
            })
        .add(
            "aten::tanh",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<TanhOperator>(device_id, node_type);
            })
        .add(
            "aten::tanh_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<TanhBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::sqrt",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SqrtOperator>(device_id, node_type);
            })
        .add(
            "aten::sqrt_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<SqrtInplaceOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::neg",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<NegOperator>(device_id, node_type);
            })
        .add(
            "aten::clamp",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ClampOperator>(device_id, node_type);
            })
        .add(
            "aten::clamp_",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ClampInplaceOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::reciprocal",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ReciprocalOperator>(device_id, node_type);
            })
        .add(
            "aten::gelu",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GeluOperator>(device_id, node_type);
            })
        .add(
            "aten::gelu_backward",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<GeluBackwardOperator>(
                  device_id, node_type);
            })
        .add(
            "aten::erf",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ErfOperator>(device_id, node_type);
            })
        .add(
            "aten::exp",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<ExpOperator>(device_id, node_type);
            })
        .add("aten::floor", [](const int device_id, c10::ScalarType node_type) {
          return std::make_shared<FloorOperator>(device_id, node_type);
        });
