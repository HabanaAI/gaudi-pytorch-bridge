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
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/resize.h"
#include "lazy_kernels.h"

using namespace torch;

namespace habana {

// Getting the HPU worker generator instance
Generator& getDefaultHPUGenerator() {
  static auto default_gen_hpu = createHPUGenerator();
  return default_gen_hpu;
}

// Utility to create a CPUGeneratorImpl. Returns a shared_ptr
Generator createHPUGenerator() {
  auto default_cpu_gen = at::detail::getDefaultCPUGenerator();
  auto gen = make_generator<CPUGeneratorImpl>(default_cpu_gen.current_seed());
  return gen;
}

uint32_t get_seed_hpu(const c10::optional<Generator>& gen) {
  CPUGeneratorImpl* generator =
      get_generator_or_default<CPUGeneratorImpl>(gen, getDefaultHPUGenerator());

  auto context = habana_lazy::get_device_lazy_execution_context();
  if (context->getDryRun()) {
    return 0;
  }
  // Acquire lock when using random generators
  std::lock_guard<std::mutex> lock(generator->mutex_);
  return generator->random();
}

at::Tensor get_seed_tensor_hpu(const c10::optional<Generator>& gen) {
  int seed = get_seed_hpu(gen);
  at::Tensor seed_tensor = at::tensor(seed);
  auto t = habana_lazy::append_to_batch_h2d_list(seed_tensor);
  auto context = habana_lazy::get_device_lazy_execution_context();
  if (context->getCapturing()) {
    habana_lazy::HbLazyTensor hb_tensor = habana_lazy::GetHbLazyTensor(t);
    hb_tensor.getDataPtr()->is_random_seed_tensor = true;
    context->getSeedTensorMap()[hb_tensor.getDataPtr()->unique_id] = gen;
  }
  return t;
}

} // namespace habana

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
