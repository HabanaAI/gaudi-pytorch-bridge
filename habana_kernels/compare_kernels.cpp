/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <synapse_api.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/script.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_helpers/logging.h"
#include "habana_helpers/logging_pt.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/passes/transform_graph.h"
#include "pytorch_helpers/habana_helpers/dtype_helpers.h"

using namespace torch;
namespace habana {

void CompareOutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input arguments for Compare Out Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg 1 for compare op needs to be tensor type");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg 2 for compare op needs to be of tensor type");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg 3 for compare op needs to be of tensor type");
  // Tensor self = inputs[0].toTensor();
  // Tensor other = inputs[1].toTensor();
  Tensor output = inputs[2].toTensor();

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

InferOutputMetaRetType CompareOutOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  auto output = inputs[2].toTensor();
  InferOutputMetaRetType out;
  out.AddOutputTensor(TensorMetaData(
      output.sizes().vec(),
      HabanaOperator::CalculateStrides(
          output.sizes().vec(), output.suggest_memory_format()),
      output.scalar_type(),
      output.suggest_memory_format()));
  return out;
}

void CompareOutWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  // this check is for stack during graph execution
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of input expected for Compare operator");
  // Note that there is no (Scalar, Tensor) version for comparison ops
  // in native_functions.yaml
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");

  std::shared_ptr<HabanaOperator> compareOp;
  if (inputs[1].isTensor()) { // Both inputs are tensors
    compareOp = make_operator<CompareOutOperator>(
        this->p_context_->device_id_, this->scalarType_, guid_);
    compareOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    compareOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    compareOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    constOp->AllocateAndAddSynapseNode(
        graph, constOp_stack, OutputMetaDataVector(1));

    compareOp = make_operator<CompareOutOperator>(
        this->p_context_->device_id_, this->scalarType_, guid_);
    compareOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    compareOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
    // replace 2nd scalar input with a tensor in stack
    auto org_input = inputs.at(1);
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, constOp->GetOutputs()[0]);
    compareOp->AllocateAndAddSynapseNode(graph, inputs, output_metadata);
    // revert the input stack changes
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(inputs.cbegin() + 1, org_input);
  }

  p_context_->pt_outputs_.emplace_back(compareOp->GetOutputs()[0]);
  synapse_helpers::tensor& out_syn_t = compareOp->GetSynOutputs()[0];
  p_context_->syn_outputs_.emplace_back(out_syn_t);
}

InferOutputMetaRetType CompareOutWrapperOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;

  std::shared_ptr<HabanaOperator> compareOp;
  if (inputs[1].isTensor()) { // Both inputs are tensors
    compareOp = make_operator<CompareOutOperator>(
        this->p_context_->device_id_, this->scalarType_, guid_);
    auto& compareOp_out = out.call_InferOutputMeta(compareOp, inputs);
    auto compareOp_out_tensor = compareOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(compareOp_out_tensor));
    return out;
  } else { // 2nd input is a scalar
    // add constant node to convert 2nd input to tensor
    auto arg1 = inputs[0].toTensor();
    auto constOp = make_operator<ConstantOperator>(
        this->p_context_->device_id_, this->scalarType_);
    auto const_shape_tensor = habana::createPTTensor(
        arg1, {1}, arg1.options(), at::MemoryFormat::Contiguous, false);
    torch::jit::Stack constOp_stack = {IValue(const_shape_tensor), inputs[1]};
    auto& constOp_out = out.call_InferOutputMeta(constOp, constOp_stack);

    // replace 2nd scalar input with a tensor in stack
    inputs.erase(inputs.cbegin() + 1);
    inputs.emplace(
        inputs.cbegin() + 1, std::get<1>(constOp_out.GetOutputTensor(0)));
    compareOp = make_operator<CompareOutOperator>(
        this->p_context_->device_id_, this->scalarType_, guid_);
    auto& compareOp_out = out.call_InferOutputMeta(compareOp, inputs);
    auto compareOp_out_tensor = compareOp_out.GetOutputTensor(0);
    out.MoveToOutput(std::move(compareOp_out_tensor));
    return out;
  }
}

void CompareOutWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  auto output = inputs[2].toTensor();
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

void CompareWrapperOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of input arguments for Compare Operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be a tensor");
  TORCH_CHECK(
      inputs[1].isTensor() || inputs[1].isScalar(),
      "Input arg2 type expected to be a tensor or scalar");
  std::vector<int64_t> out_shape;
  Tensor operand;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape =
        compute_output_shape(inputs[0].toTensor(), inputs[1].toTensor());
  } else if (inputs[0].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape = operand.sizes().vec();
  } else {
    operand = inputs[1].toTensor();
    out_shape = operand.sizes().vec();
  }
  auto output = habana::createPTTensor(
      operand,
      IntArrayRef(out_shape.data(), out_shape.size()),
      operand.options(),
      operand.suggest_memory_format(),
      c10::ScalarType::Bool,
      output_metadata.at(0).persistent);
  inputs.push_back(output);
  CompareOutWrapperOperator::AllocateAndAddSynapseNode(
      graph, inputs, output_metadata);
  // revert input stack
  inputs.pop_back();
}

InferOutputMetaRetType CompareWrapperOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  std::vector<int64_t> out_shape;
  Tensor operand;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape =
        compute_output_shape(inputs[0].toTensor(), inputs[1].toTensor());
  } else if (inputs[0].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape = operand.sizes().vec();
  } else {
    operand = inputs[1].toTensor();
    out_shape = operand.sizes().vec();
  }
  auto output = habana::createPTTensor(
      operand,
      IntArrayRef(out_shape.data(), out_shape.size()),
      operand.options(),
      operand.suggest_memory_format(),
      c10::ScalarType::Bool,
      false);
  inputs.push_back(output);
  return CompareOutWrapperOperator::InferOutputMeta(inputs);
}

void CompareWrapperOperator::SetPTOutputs(torch::jit::Stack& inputs) {
  std::vector<int64_t> out_shape;
  Tensor operand;
  if (inputs[0].isTensor() && inputs[1].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape =
        compute_output_shape(inputs[0].toTensor(), inputs[1].toTensor());
  } else if (inputs[0].isTensor()) {
    operand = inputs[0].toTensor();
    out_shape = operand.sizes().vec();
  } else {
    operand = inputs[1].toTensor();
    out_shape = operand.sizes().vec();
  }
  auto output = habana::createPTTensor(
      operand,
      IntArrayRef(out_shape.data(), out_shape.size()),
      operand.options(),
      operand.suggest_memory_format(),
      c10::ScalarType::Bool,
      true);
  std::vector<at::Tensor> v{output};
  HabanaOperator::SetPTOutputs(v);
}

std::vector<int64_t> CompareWrapperOperator::compute_output_shape(
    const Tensor& arg1,
    const Tensor& arg2) {
  auto out_size = habana_helpers::compute_broadcast_shape(arg1, arg2);
  return out_size;
}

} // namespace habana
