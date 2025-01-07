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
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>

#include "backend/create_pt_tensor.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/random_gen_kernels.h"

using namespace torch;

using namespace habana;

InferOutputMetaRetType RandomShuffleOperator::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto self = inputs[0].toTensor();
  out.AddOutputTensor(TensorMetaData(
      self.sizes().vec(),
      HabanaOperator::CalculateStrides(
          self.sizes(), self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format()));
  return out;
}

/************************************************************************
 * @brief This function implements synapse node addition for random_shuffle
 * function with 2 input arguments (where all arguments are tensors)
 ************************************************************************/
void RandomShuffleOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of input expected for random shuffle operator");
  TORCH_CHECK(
      inputs[0].isTensor(), "Input condition type expected to be a tensor");

  auto self = inputs[0].toTensor();

  auto output =
      at::empty(self.sizes(), self.options(), self.suggest_memory_format());

  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, nullptr, 0);
}

InferOutputMetaRetType RandpermOperatorHT::InferOutputMeta(
    torch::jit::Stack& inputs) {
  InferOutputMetaRetType out;
  auto host_tensor = inputs[0].toTensor();
  auto output = inputs[2].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana::createPTTensor(output, false);

  auto arangeOp = make_operator<ArangeOperatorHT>(
      this->p_context_->device_id_, scalar_type);
  torch::jit::Stack stack{
      IValue(host_tensor), IValue(arangeOutput), IValue(output)};
  out.call_InferOutputMeta(arangeOp, stack);

  stack.clear();
  stack.emplace_back(IValue(arangeOutput));
  auto randShuffleOp = make_operator<RandomShuffleOperator>(
      this->p_context_->device_id_, at::ScalarType::Int);

  auto& randShuffle_op_out = out.call_InferOutputMeta(randShuffleOp, stack);
  auto randShuffle_op_tensor = randShuffle_op_out.GetOutputTensor()[0];

  out.MoveToOutput(std::move(randShuffle_op_tensor));

  return out;
}

void RandpermOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size",
      inputs.size(),
      " of inputs expected for RandpermOperatorHT");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg0 expected to be Tensor for RandpermOperatorHT");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg1 expected to be Tensor for RandpermOperatorHT");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg2 expected to be Tensor for RandpermOperatorHT");

  auto host_tensor = inputs[0].toTensor();
  auto output = inputs[2].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana::createPTTensor(output, false);
  auto arangeOp = make_operator<ArangeOperatorHT>(
      this->p_context_->device_id_, scalar_type);
  arangeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  arangeOp->AllocateSynapseInput(graph, arangeOutput, false);
  torch::jit::Stack stack{
      IValue(host_tensor), IValue(arangeOutput), IValue(output)};
  arangeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Create RandomShuffle operator in int32 precision which is the only
  // supported by the tpc_kernel. If the inputs have different dtype, they will
  // be automatically casted to int32 dtype.
  auto randShuffleOp = make_operator<RandomShuffleOperator>(
      this->p_context_->device_id_, at::ScalarType::Int);
  stack.emplace_back(IValue(arangeOutput));
  randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
  randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[1]);
  randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  p_context_->syn_outputs_.emplace_back(
      std::move(randShuffleOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(
      std::move(randShuffleOp->GetOutputs()[0]));
}

void RandpermOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size",
      inputs.size(),
      " of inputs expected for Randperm Operator");
  TORCH_CHECK(
      inputs[0].isScalar(),
      "Input arg0 expected to be Scalar for RandpermOperator operator");
  TORCH_CHECK(
      inputs[1].isTensor(),
      "Input arg1 expected to be (seed) Tensor for RandpermOperator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg2 expected to be Tensor for RandpermOperator Operator");

  auto seed_tensor = inputs[1].toTensor();
  auto output = inputs[2].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana::createPTTensor(output, false);
  auto arangeOp =
      make_operator<ArangeOperator>(this->p_context_->device_id_, scalar_type);
  // Order of tensors
  // {seed_tensor, output_tensor}
  auto n = inputs[0].toInt();
  auto start = 0;
  auto end = n;
  auto step = 1;
  arangeOp->AllocateSynapseInput(graph, arangeOutput, false);
  torch::jit::Stack stack{
      IValue(start), IValue(end), IValue(step), IValue(arangeOutput)};
  arangeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Create RandomShuffle operator in int32 precision which is the only
  // supported by the tpc_kernel. If the inputs have different dtype, they will
  // be automatically casted to int32 dtype.
  auto randShuffleOp = make_operator<RandomShuffleOperator>(
      this->p_context_->device_id_, at::ScalarType::Int);
  stack.emplace_back(IValue(arangeOutput));
  randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
  randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  p_context_->syn_outputs_.emplace_back(
      std::move(randShuffleOp->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(
      std::move(randShuffleOp->GetOutputs()[0]));
}

void HabanaRandomSeedOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const habana::OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 1,
      "Incorrect size of inputs expected for HabanaRandomSeedOperator operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for HabanaRandomSeedOperator operator");

  Tensor input = inputs[0].toTensor();
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Int,
      "Input arg1.dtype expected to be Int for HabanaRandomSeedOperator operator");

  const auto is_output_persistent = output_metadata.at(0).persistent;

  synapse_helpers::tensor& input_syn_tensor = p_context_->syn_inputs_[0];
  std::vector<synTensor> syn_inputs;
  syn_inputs.push_back(input_syn_tensor.get());

  // Add random_seed_u32 node to graph
  auto input_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      kernel_meta_data_.synapse_input_layout);

  auto guid = "random_seed_u32";

  graph.add_node(
      std::move(syn_inputs),
      {},
      nullptr,
      0,
      guid,
      nullptr,
      input_layouts.data(),
      nullptr,
      false,
      getContextHints());

  auto output = habana::createPTTensor(
      input,
      input.sizes(),
      input.options(),
      input.suggest_memory_format(),
      is_output_persistent);

  // Create synapse output tensor
  AllocateSynapseOutput(
      graph,
      output,
      output_metadata.at(0),
      false); // is_shape_tensor

  synapse_helpers::tensor& output_syn_tensor = p_context_->syn_outputs_[0];
  std::vector<synTensor> syn_outputs{output_syn_tensor.get()};

  auto output_layouts = synapse_helpers::layouts::getSynapseLayoutFormat(
      kernel_meta_data_.synapse_output_layout);

  // Add random_seed_u32 node to graph
  guid = "identity";

  graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      guid,
      nullptr,
      input_layouts.data(),
      output_layouts.data(),
      false,
      getContextHints());
}

static auto& RandomGenKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::randperm_out", KERNEL_FN(RandpermOperator))
        .add("hpu::randperm_out_ds_ht", KERNEL_FN(RandpermOperatorHT))
        .add("hpu::habana_random_seed", KERNEL_FN(HabanaRandomSeedOperator));
