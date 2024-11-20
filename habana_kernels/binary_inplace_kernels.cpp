/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#include <ATen/ExpandUtils.h>
#include <torch/script.h>
#include <memory>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"

using namespace torch;

/************************************************************************
 * @brief This function implements synapse node addition for
 * binary operators where 2 inputs are tensors & 3rd input is a scalar.
 * Mismatch in input tensor dims is taken care of using reshape nodes,
 * whereas scalar to tensor conversion is done using constant node.
 ************************************************************************/
void habana::BinaryInplaceOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 0 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input 1 type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  if (inputs[2].toScalar().toFloat() != 1.0) {
    // Multiplication between arg2 and alpha is required
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, this->scalarType_);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack mulOp_stack{inputs[1], inputs[2]};

    mulOp->AllocateAndAddSynapseNode(
        graph, mulOp_stack, habana::OutputMetaDataVector(1));
    synapse_helpers::tensor_or_ref& mulOp_out = mulOp->GetSynOutputs()[0];

    // Note here we are using input[0] to store output[0]
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(arg1);

    synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor = mulOp_out;

    std::vector<synTensor> syn_inputs{
        arg1_syn_tensor.get(), arg2_syn_tensor.get()};

    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic,
        getContextHints());
  } else {
    // Note here we are using input[0] to store output[0]
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(arg1);
    synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

    std::vector<synTensor> syn_inputs{
        arg1_syn_tensor.get(), arg2_syn_tensor.get()};

    synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
    std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

    graph.add_node(
        std::move(syn_inputs),
        std::move(syn_outputs),
        nullptr,
        0,
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic,
        getContextHints());
  }
}

habana::InferOutputMetaRetType habana::BinaryInplaceOperatorWithAlpha::
    InferOutputMeta(torch::jit::Stack& inputs) {
  Tensor arg1 = inputs[0].toTensor();

  InferOutputMetaRetType out;
  if (inputs[2].toScalar().toFloat() != 1.0) {
    // Multiplication between arg2 and alpha is required
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, this->scalarType_);
    torch::jit::Stack mulOp_stack{inputs[1], inputs[2]};
    auto& mulOp_out = out.call_InferOutputMeta(mulOp, mulOp_stack);
    auto out_tensor = mulOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  }

  out.AddOutputTensor(TensorMetaData(
      arg1.sizes().vec(),
      HabanaOperator::CalculateStrides(
          arg1.sizes(), arg1.suggest_memory_format()),
      arg1.scalar_type(),
      arg1.suggest_memory_format()));
  return out;
}

/************************************************************************
 * @brief This function implements synapse node addition for Binary OPs
 * with 3 input arguments (where 1st input is always a tensor whereas
 * 2nd and 3rd inputs can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryInplaceWrapperOperatorWithAlpha::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3, "Incorrect size of input expected for add operator");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[1].isTensor(),
      "At least one of the inputs arg1 or arg2 expected to be a tensor");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[0].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isScalar(), "Input arg3 type expected to be scalar");

  auto binaryOp = make_operator<BinaryInplaceOperatorWithAlpha>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() &&
      inputs[1].isTensor()) { // First 2 inputs are both tensors
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);

  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add node to convert scalar to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    binaryOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    binaryOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }

  p_context_->pt_outputs_.emplace_back(binaryOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryOp->GetSynOutputs()[0]));
}

habana::InferOutputMetaRetType habana::BinaryInplaceWrapperOperatorWithAlpha::
    InferOutputMeta(torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto binaryOp = make_operator<BinaryInplaceOperatorWithAlpha>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() &&
      inputs[1].isTensor()) { // First 2 inputs are both tensors
    auto& binaryOp_out = out.call_InferOutputMeta(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    auto& constOp_out = out.call_InferOutputMeta(constOp, constOp_stack);
    auto const_out_tensor = constOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(const_out_tensor));

    auto& binaryOp_out = out.call_InferOutputMeta(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
  }
  return out;
}

/************************************************************************
 * @brief This function implements synapse node addition for
 * inplace binary operators where both inputs are tensors. Mismatch in
 * input tensor dims is also taken care of using reshape nodes.
 ************************************************************************/
void habana::BinaryInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // this check is for stack during graph execution
  static_cast<void>(output_metadata);
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  Tensor arg1 = inputs[0].toTensor();
  Tensor arg2 = inputs[1].toTensor();

  // Note here we are using input[0] to store output[0]
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(arg1);

  synapse_helpers::tensor& arg1_syn_tensor = p_context_->syn_inputs_[0];
  synapse_helpers::tensor& arg2_syn_tensor = p_context_->syn_inputs_[1];

  std::vector<synTensor> syn_inputs{
      arg1_syn_tensor.get(), arg2_syn_tensor.get()};

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      guid_,
      nullptr,
      nullptr,
      nullptr,
      deterministic,
      getContextHints());
}

habana::InferOutputMetaRetType habana::BinaryInplaceOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  Tensor arg1 = inputs[0].toTensor();

  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      arg1.sizes().vec(),
      HabanaOperator::CalculateStrides(
          arg1.sizes(), arg1.suggest_memory_format()),
      arg1.scalar_type(),
      arg1.suggest_memory_format()));
  return out;
}

/************************************************************************
 * @brief This function implements synapse node addition for inplace
 * Binary OPs with 2 input arguments (where 1st input is always a
 * tensor whereas 2nd input can be a tensor or a scalar)
 ************************************************************************/
void habana::BinaryInplaceWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input expected for Binary operator");
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[1].isTensor(),
      "At least one of the inputs arg1 or arg2 expected to be a tensor");
  // Note that pow has a (Scalar, Tensor) variant in native_functions.yaml
  // although mul and div do not
  TORCH_CHECK(
      inputs[0].isTensor() || inputs[0].isScalar(),
      "Input arg1 type expected to be a tensor or scalar");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");

  auto binaryInplaceOp = make_operator<BinaryInplaceOperator>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    binaryInplaceOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryInplaceOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    binaryInplaceOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, habana::OutputMetaDataVector(1));
    binaryInplaceOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    binaryInplaceOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace input scalar with input tensor in the stack
    inputs.pop_back();
    inputs.emplace_back(constOp->GetOutputs()[0]);
    binaryInplaceOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  }

  p_context_->pt_outputs_.emplace_back(binaryInplaceOp->GetOutputs()[0]);
  p_context_->syn_outputs_.emplace_back(
      std::move(binaryInplaceOp->GetSynOutputs()[0]));
}

habana::InferOutputMetaRetType habana::BinaryInplaceWrapperOperator::
    InferOutputMeta(torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto binaryOp = make_operator<BinaryInplaceOperator>(
      this->p_context_->device_id_, guid_, this->scalarType_);

  if (inputs[0].isTensor() && inputs[1].isTensor()) { // Both inputs are tensors
    auto& binaryOp_out = out.call_InferOutputMeta(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else if (inputs[0].isTensor() && inputs[1].isScalar()) { // 2nd input is a
                                                             // scalar
    auto arg1 = inputs[0].toTensor();
    // add constant node to convert 2nd input to tensor
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    auto& constOp_out = out.call_InferOutputMeta(constOp, constOp_stack);
    auto const_out_tensor = constOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(const_out_tensor));

    auto& binaryOp_out = out.call_InferOutputMeta(binaryOp, inputs);
    auto out_tensor = binaryOp_out.GetOutputTensor(0);
  }
  return out;
}

void habana::AddcmulInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Addcmul operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be tensor for Addcmul operator");
  TORCH_CHECK(
      inputs[3].isScalar(),
      "Input arg4 expected to be Scalar for Addcmul operator");

  auto self = inputs[0].toTensor();
  auto tensor1 = inputs[1].toTensor();
  auto tensor2 = inputs[2].toTensor();
  auto alphaValue = inputs[3].toScalar();

  std::vector<c10::IValue> stack;
  at::ScalarType scalar_type = self.scalar_type();

  // special handling required. synapse cannot handle same tensor given as both
  // inputs to a binary op
  if (tensor1.is_same(tensor2)) {
    // x^2 implemented as x*x. Identity node used to create aliased tensor
    // since GC/TPC does not like giving same tensor as both inputs to a
    // binary op
    auto identityOp = make_operator<IdentityOperator>(
        this->p_context_->device_id_, scalar_type);
    identityOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(tensor1)};
    identityOp->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, scalar_type);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulOp->SetSynapseInput(identityOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(tensor1));
    stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
    mulOp->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    // Create Add operator
    auto addOp = make_operator<habana::AddInplaceOperator>(
        this->p_context_->device_id_, scalar_type);
    addOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    addOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue));
    addOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(addOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(addOp->GetOutputs()[0]));
    p_context_->syn_inputs_.pop_back();
  } else {
    // Create Mul operator
    auto mulOp = make_operator<habana::MulOperator>(
        this->p_context_->device_id_, scalar_type);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulOp->SetSynapseInput(p_context_->syn_inputs_[2]);
    stack.emplace_back(IValue(tensor1));
    stack.emplace_back(IValue(tensor2));
    mulOp->AllocateAndAddSynapseNode(
        graph, stack, habana::OutputMetaDataVector(1));
    stack.clear();

    // Create Add operator
    auto addOp = make_operator<habana::AddInplaceOperator>(
        this->p_context_->device_id_, scalar_type);
    addOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    addOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue));
    addOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(std::move(addOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(addOp->GetOutputs()[0]));
  }
}

static auto& BinaryInplaceKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::add_.Tensor", KERNEL_FN(AddInplaceOperator))
        .add("hpu::add_.Scalar", KERNEL_FN(AddInplaceOperator))
        .add("aten::add_.Tensor", KERNEL_FN(AddInplaceOperator))
        .add("aten::add_.Scalar", KERNEL_FN(AddInplaceOperator));
