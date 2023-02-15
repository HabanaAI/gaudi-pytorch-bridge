/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <perf_lib_layer_params.h>
#include <torch/script.h>

#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/WrapDimUtils.h>

#include "backend/create_pt_tensor.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/unary_kernels.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"

using namespace torch;
using namespace torch::jit;
using namespace habana;

void UnaryOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1, "Incorrect size of inputs expected for operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  at::Tensor input = inputs[0].toTensor();
  auto output = habana::createPTTensor(input, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

OutputShapeInfRetType UnaryOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
  auto input = inputs[0].toTensor();
  out.AddOutputTensor(TensorMetaData(
      input.sizes().vec(),
      HabanaOperator::CalculateStrides(
          input.sizes().vec(), input.suggest_memory_format()),
      input.scalar_type(),
      input.suggest_memory_format()));
  return out;
}

void UnaryInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  static_cast<void>(output_metadata);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
  p_context_->pt_outputs_.emplace_back(inputs[0].toTensor());
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void UnaryLikeOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  if (m_inplace) {
    p_context_->syn_outputs_.emplace_back(
        habana_helpers::duplicate_tensor_in_memory_section(
            p_context_->syn_inputs_[0], graph, output_metadata.at(0).external));
    p_context_->pt_outputs_.emplace_back(inputs[0].toTensor());
  } else {
    auto output = habana::createPTTensor(
        inputs[0].toTensor(), output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
  }
}

void UnaryBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
      input.sizes());

  auto grad_output =
      habana::createPTTensor(input, output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, grad_output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void EluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      std::string("Incorrect size of inputs expected for ") +
          (m_inplace ? "Elu_" : "Elu") + " operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input 1 type expected to be a tensor");
  TORCH_CHECK(inputs[1].isScalar(), "Input 2 type expected to be a scalar");
  TORCH_CHECK(inputs[2].isScalar(), "Input 3 type expected to be a scalar");
  TORCH_CHECK(inputs[3].isScalar(), "Input 4 type expected to be a scalar");

  if (inputs[2].toScalar().toFloat() != 1. or
      inputs[3].toScalar().toFloat() != 1.) {
    PT_KERNEL_WARN("Elu supports scale and input_scale as 1.");
  }

  ns_EluKernel::Params param{inputs[1].toScalar().toFloat()};

  UnaryLikeOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void GeluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      (inputs.size() == 1 || inputs.size() == 2),
      "Incorrect size of inputs expected for Gelu operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu operator");

  auto self = inputs[0].toTensor();

  auto output1 = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);

  // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
  // In graph mode we want 2nd output to be non-persistent to reduce memory
  // consumption
  auto output2 = habana::createPTTensor(
      self, self.sizes(), self.options(), self.suggest_memory_format(), false);

  std::vector<at::Tensor> outputs{output1, output2};
  AllocateSynapseOutputs(
      graph, outputs, {output_metadata.at(0), OutputMetaData()});
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void HbGeluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Gelu operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Gelu operator");

  auto self = inputs[0].toTensor();

  auto output1 = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(0).persistent);

  // TPC kernel expects two outputs first is gelu_fwd second output is tanhz
  // In graph mode we want 2nd output to be non-persistent to reduce memory
  // consumption
  auto output2 = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      output_metadata.at(1).persistent);

  std::vector<at::Tensor> outputs{output1, output2};
  AllocateSynapseOutputs(graph, outputs, output_metadata);
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void GeluBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2 || inputs.size() == 3,
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
  auto is_sv = false;
  if (inputs.size() == 3)
    is_sv = !inputs[2].isTensor();
  if (inputs.size() == 2 || is_sv) {
    // x^3 implemented as x*x*x. Identity node used to create aliased tensor
    // since GC/TPC does not like giving same tensor as both inputs to a
    // binary op
    auto identityOp = make_operator<IdentityOperator>(
        this->p_context_->device_id_, scalar_type);
    identityOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    torch::jit::Stack stack = {IValue(self)};
    identityOp->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mulpow1Op =
        make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
    mulpow1Op->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulpow1Op->SetSynapseInput(identityOp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(identityOp->GetOutputs()[0]));
    mulpow1Op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mulpow2Op =
        make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
    mulpow2Op->SetSynapseInput(p_context_->syn_inputs_[1]);
    mulpow2Op->SetSynapseInput(mulpow1Op->GetSynOutputs()[0]);
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulpow1Op->GetOutputs()[0]));
    mulpow2Op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // Create Add operator
    auto addOp =
        make_operator<AddOperator>(this->p_context_->device_id_, scalar_type);
    addOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    addOp->SetSynapseInput(mulpow2Op->GetSynOutputs()[0]);
    // Build Params for the graph
    Scalar alphaValue = 0.044715;
    stack.emplace_back(IValue(self));
    stack.emplace_back(IValue(mulpow2Op->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue));
    addOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // Create Mul operator
    auto mulOp =
        make_operator<MulOperator>(this->p_context_->device_id_, scalar_type);
    mulOp->SetSynapseInput(addOp->GetSynOutputs()[0]);
    // Build Params for the graph
    Scalar alphaValue_2 = M_2_SQRTPI * M_SQRT1_2;
    stack.emplace_back(IValue(addOp->GetOutputs()[0]));
    stack.emplace_back(IValue(alphaValue_2));
    mulOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // Create Tanh operator
    auto tanhOp =
        make_operator<TanhOperator>(this->p_context_->device_id_, scalar_type);
    tanhOp->SetSynapseInput(mulOp->GetSynOutputs()[0]);
    // Build Params for the graph
    stack.emplace_back(IValue(mulOp->GetOutputs()[0]));
    tanhOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto output = habana::createPTTensor(
        self,
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    synapse_helpers::tensor& synInput0 = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& synInput1 = p_context_->syn_inputs_[1];
    synapse_helpers::tensor& synInput2 = tanhOp->GetSynOutputs()[0];

    std::vector<synTensor> syn_in{
        synInput0.get(), synInput1.get(), synInput2.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        nullptr,
        0,
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  } else {
    auto output = habana::createPTTensor(
        self,
        self.sizes(),
        self.options(),
        self.suggest_memory_format(),
        output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    synapse_helpers::tensor& synInput0 = p_context_->syn_inputs_[0];
    synapse_helpers::tensor& synInput1 = p_context_->syn_inputs_[1];
    synapse_helpers::tensor& synInput2 = p_context_->syn_inputs_[2];

    std::vector<synTensor> syn_in{
        synInput0.get(), synInput1.get(), synInput2.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        nullptr,
        0,
        guid_,
        nullptr,
        nullptr,
        nullptr,
        deterministic);
  }
}

void ReciprocalOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for Reciprocal operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Reciprocal operator");

  auto self = inputs[0].toTensor();
  auto result = habana::createPTTensor(self, output_metadata.at(0).persistent);
  inputs.insert(inputs.begin(), IValue(result));

  ReciprocalOutOperator::AllocateAndAddSynapseNode(
      graph, inputs, output_metadata);
}

void ReciprocalOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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

  AllocateSynapseOutput(graph, result, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void ClampOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
  param.upperBound.f = max.has_value() ? max.value().to<float>()
                                       : std::numeric_limits<float>::max();
  param.lowerBound.f = min.has_value() ? min.value().to<float>()
                                       : -std::numeric_limits<float>::max();

  if (self.scalar_type() == c10::ScalarType::Int) {
    // Guid needs to be updated since TPC only supports F32/BF16
    SetGuid("clamp_fwd_f32");
    // Cast Input tensor to Float tensor
    std::string node_type = "cast_i32_to_f32";

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
    stack.clear();

    AllocateSynapseOutput(graph, output_float, OutputMetaData());
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    node_type = "clamp_fwd_f32";
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        std::move(node_type),
        nullptr,
        nullptr,
        nullptr,
        deterministic);

    node_type = "cast_f32_to_i32";
    // Create cast operator
    auto floatToIntOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    floatToIntOp->SetSynapseInput(p_context_->syn_outputs_[0]);

    // Build Params for the graph
    stack = {IValue(p_context_->pt_outputs_[0]), IValue(c10::ScalarType::Int)};

    floatToIntOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_[0] = std::move(floatToIntOp->GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(floatToIntOp->GetOutputs()[0]);

  } else {
    auto output =
        habana::createPTTensor(self, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  }
}

void ClampMinOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Clamp operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  auto self = inputs[0].toTensor();
  auto min = inputs[1].isScalar() ? inputs[1].toScalar()
                                  : inputs[1].toOptional<Scalar>();

  ns_ClampKernel::Params param;
  param.upperBound.f = std::numeric_limits<float>::max();
  param.lowerBound.f = min.has_value() ? min.value().to<float>()
                                       : -std::numeric_limits<float>::max();

  if (self.scalar_type() == c10::ScalarType::Int) {
    // Guid needs to be updated since TPC only supports F32/BF16
    SetGuid("clamp_fwd_f32");
    // Cast Input tensor to Float tensor
    std::string node_type = "cast_i32_to_f32";

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
    stack.clear();

    AllocateSynapseOutput(graph, output_float, OutputMetaData());
    synapse_helpers::tensor& synOutput = p_context_->syn_outputs_[0];

    std::vector<synTensor> syn_in{float_syn_tensor.get()};
    std::vector<synTensor> syn_out{synOutput.get()};

    node_type = "clamp_fwd_f32";
    graph.add_node(
        std::move(syn_in),
        std::move(syn_out),
        &param,
        sizeof(param),
        std::move(node_type),
        nullptr,
        nullptr,
        nullptr,
        deterministic);

    node_type = "cast_f32_to_i32";
    // Create cast operator
    auto floatToIntOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    floatToIntOp->SetSynapseInput(p_context_->syn_outputs_[0]);
    // Build Params for the graph
    stack = {IValue(p_context_->pt_outputs_[0]), IValue(c10::ScalarType::Int)};

    floatToIntOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_[0] = std::move(floatToIntOp->GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(floatToIntOp->GetOutputs()[0]);

  } else {
    auto output =
        habana::createPTTensor(self, output_metadata.at(0).persistent);
    AllocateSynapseOutput(graph, output, output_metadata.at(0));
    AddNodeToSynapseGraph(graph, &param, sizeof(param));
  }
}

void ClampInplaceOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
  param.upperBound.f = max.has_value() ? max.value().to<float>()
                                       : std::numeric_limits<float>::max();
  param.lowerBound.f = min.has_value() ? min.value().to<float>()
                                       : -std::numeric_limits<float>::max();
  if (p_context_->pt_inputs_.size() == 0)
    p_context_->pt_inputs_.emplace_back(inputs[0].toTensor());
  AllocateSynapseInplaceOutput(graph, output_metadata.at(0).external);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void HardsigmoidOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  const unsigned short constExpectedNoOfInput = 1;
  // hardsigmoid (x) = 0          if x <= -3
  //                   1          if x >= +3
  //                   x/6 + 1/2  otherwise
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;

  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      std::string("Expected ") + std::to_string(constExpectedNoOfInput) +
          " input for " +
          (m_inplace ? "HardsigmoidOperator_" : "HardsigmoidOperator") +
          " operator but received " + std::to_string(inputs.size()) +
          " inputs.");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");

  ns_HardSigmoidKernel::Params param{alpha, beta};
  UnaryLikeOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void HardsigmoidBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  const unsigned short constExpectedNoOfInput = 2;
  constexpr float alpha = 1 / 6.0f;
  constexpr float beta = 1 / 2.0f;
  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      std::string("Expected ") + std::to_string(constExpectedNoOfInput) +
          " inputs for HardsigmoidOperator operator" + " but received " +
          std::to_string(inputs.size()) + " inputs.");
  ns_HardSigmoidKernel::Params param{alpha, beta};
  auto output = habana::createPTTensor(
      inputs[0].toTensor(), output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

void SiluOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  const int64_t correctInputSize = 1;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size " + std::to_string(inputs.size()) +
          " provided as input, while expected size is " +
          std::to_string(correctInputSize) + " for SiluOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for SiluOperator");

  auto self = inputs[0].toTensor();
  at::ScalarType scalar_type = self.scalar_type();
  size_t device_id = self.device().index();

  SigmoidOperator Op(device_id, scalar_type);

  Op.SetSynapseInput(p_context_->syn_inputs_[0]);

  std::vector<c10::IValue> stack{IValue(self)};
  Op.AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  stack.clear();

  // Create Mul operator
  MulOperator mulOp(this->p_context_->device_id_, scalar_type);
  mulOp.SetSynapseInput(p_context_->syn_inputs_[0]);
  mulOp.SetSynapseInput(Op.GetSynOutputs()[0]);

  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(Op.GetOutputs()[0]));

  mulOp.AllocateAndAddSynapseNode(graph, stack, output_metadata);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(std::move(mulOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(std::move(mulOp.GetOutputs()[0]));
}

OutputShapeInfRetType SiluBackwardOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  int out_index = inputs.size() - 1;
  auto output = inputs[out_index].toTensor();
  OutputShapeInfRetType out;
  out.AddOutputTensor(TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

void SiluBackwardOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  const unsigned short constExpectedNoOfInput = 2;

  TORCH_CHECK(
      inputs.size() == constExpectedNoOfInput,
      "Expected ",
      constExpectedNoOfInput,
      " inputs for SiluBackwardOperator but received ",
      inputs.size(),
      " inputs.");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for SiluBackwardOperator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg2 expected to be Tensor for SiluBackwardOperator");

  auto grad = inputs[0].toTensor();
  auto self = inputs[1].toTensor();

  auto scalar_type = self.scalar_type();

  // Sigmoid
  SigmoidOperator sigmoidOp(this->p_context_->device_id_, scalar_type);
  sigmoidOp.SetSynapseInput(p_context_->syn_inputs_[1]);
  torch::jit::Stack stack = {IValue(self)};
  sigmoidOp.AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Do G*S
  habana::MulOperator mulOp(this->p_context_->device_id_, scalar_type);

  mulOp.SetSynapseInput(p_context_->syn_inputs_[0]); // grad
  mulOp.SetSynapseInput(sigmoidOp.GetSynOutputs()[0]); // sigmoid

  stack.emplace_back(IValue(grad));
  stack.emplace_back(IValue(sigmoidOp.GetOutputs()[0]));
  mulOp.AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  stack.clear();

  // Do G*S*Self
  habana::MulOperator mulOp1(this->p_context_->device_id_, scalar_type);

  mulOp1.SetSynapseInput(p_context_->syn_inputs_[1]); // self
  mulOp1.SetSynapseInput(mulOp.GetSynOutputs()[0]); // sigmoid

  stack.emplace_back(IValue(self));
  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  mulOp1.AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  stack.clear();

  // Do G*S*Self*S
  habana::MulOperator mulOp2(this->p_context_->device_id_, scalar_type);

  mulOp2.SetSynapseInput(sigmoidOp.GetSynOutputs()[0]); // sigmoid
  mulOp2.SetSynapseInput(mulOp1.GetSynOutputs()[0]); // G*S*Self

  stack.emplace_back(IValue(sigmoidOp.GetOutputs()[0]));
  stack.emplace_back(IValue(mulOp1.GetOutputs()[0]));
  mulOp2.AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  stack.clear();

  // Do G*S + G*self*S
  habana::AddOperator addOp(this->p_context_->device_id_, scalar_type);

  addOp.SetSynapseInput(mulOp.GetSynOutputs()[0]); // G*S
  addOp.SetSynapseInput(mulOp1.GetSynOutputs()[0]); // G*self*S

  stack.emplace_back(IValue(mulOp.GetOutputs()[0]));
  stack.emplace_back(IValue(mulOp1.GetOutputs()[0]));
  stack.emplace_back(IValue(1.0));
  addOp.AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));

  stack.clear();

  // Do G*S + G*self*S - G*self*S*S
  habana::SubOperator subOp(this->p_context_->device_id_, scalar_type);

  subOp.SetSynapseInput(addOp.GetSynOutputs()[0]); // G*S + G*self*S
  subOp.SetSynapseInput(mulOp2.GetSynOutputs()[0]); // G*self*S*S

  stack.emplace_back(IValue(addOp.GetOutputs()[0]));
  stack.emplace_back(IValue(mulOp2.GetOutputs()[0]));
  stack.emplace_back(IValue(1.0));
  subOp.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  stack.clear();

  // Output
  p_context_->syn_outputs_.emplace_back(std::move(subOp.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(subOp.GetOutputs()[0]);
}

void IsnanOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  const int64_t correctInputSize = 1;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size " + std::to_string(inputs.size()) +
          " provided as input, while expected size is " +
          std::to_string(correctInputSize) + " for IsnanOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for IsnanOperator");

  auto self = inputs[0].toTensor();

  Tensor output = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Bool,
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

void CumsumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  const int64_t correctInputSize = 3;
  TORCH_CHECK(
      inputs.size() == correctInputSize,
      "Incorrect size ",
      inputs.size(),
      " provided as input, while expected size is ",
      correctInputSize,
      " for IsnanOperator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for CumsumOperator");
  TORCH_CHECK(
      inputs[1].isInt(), "Input arg2 expected to be Int for CumsumOperator");

  auto self = inputs[0].toTensor();
  auto dim = inputs[1].toInt();
  // TODO::Handle 3rd arument dtype, in lazy mode
  dim = at::maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);

  int tpcAxis = (self.sizes().vec().size() - 1) - dim;

  ns_CumSumKernel::Params param{
      tpcAxis, // cumsum along this
      0, // 0 =>inclusive
      0 // 0=> no reverse
  };

  UnaryLikeOperator::AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  AddNodeToSynapseGraph(graph, &param, sizeof(param));
}

static auto& UnaryKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::hbgelu2", KERNEL_FN(HbGeluOperator))
        .add("aten::hbgelu2_backward", KERNEL_FN(GeluBackwardOperator));
