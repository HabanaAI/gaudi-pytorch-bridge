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

static auto& UnaryKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::hbgelu2", KERNEL_FN(HbGeluOperator))
        .add("aten::hbgelu2_backward", KERNEL_FN(GeluBackwardOperator));
