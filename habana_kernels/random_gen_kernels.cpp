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

#include <ATen/ExpandUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>

#include "backend/create_pt_tensor.h"
#include "backend/helpers/graph.h"
#include "backend/helpers/tensor_utils.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
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

  // Acquire lock when using random generators
  std::lock_guard<std::mutex> lock(generator->mutex_);
  return generator->random();
}

at::Tensor get_seed_tensor_hpu(const c10::optional<Generator>& gen) {
  int seed = get_seed_hpu(gen);
  at::Tensor seed_tensor = at::tensor(seed);
  auto t = habana_lazy::append_to_batch_h2d_list(seed_tensor);
  auto context = habana_lazy::habana_lazy_executor.getDeviceExecutionContext(0);
  if (context->getCapturing()) {
    habana_lazy::HbLazyTensor hb_tensor = habana_lazy::GetHbLazyTensor(t);
    hb_tensor.getDataPtr()->is_random_seed_tensor = true;
    context->getSeedTensorMap()[hb_tensor.getDataPtr()->unique_id] = gen;
  }
  return t;
}

} // namespace habana

using namespace habana;

void UniformOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Uniform Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Uniform Operator");
  TORCH_CHECK(
      inputs[1].isDouble(),
      "Input arg2 expected to be Double for Uniform Operator");
  TORCH_CHECK(
      inputs[2].isDouble(),
      "Input arg3 expected to be of type Double for Uniform Operator");
  // For graph mode arg4 should be of type None
  TORCH_CHECK(
      inputs[3].isInt() || inputs[3].isNone(),
      "Input arg4 expected to be Int or None for Uniform Operator");

  auto self = inputs[0].toTensor();
  auto from = inputs[1].toDouble();
  auto to = inputs[2].toDouble();

  ns_RandomUniform::Params params;
  params.low = static_cast<float>(from);
  params.high = static_cast<float>(to);

  if (inputs[3].isNone()) {
    params.seed = get_seed_hpu(c10::nullopt);
  } else {
    auto seed = inputs[3].toInt();
    params.seed = seed;
  }

  p_context_->params_.emplace<ns_RandomUniform::Params>(params);
  p_context_->params_size_ = sizeof(params);

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, self);
  }

  AllocateSynapseOutput(graph, self, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

OutputShapeInfRetType RandomShuffleOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
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

OutputShapeInfRetType RandpermOperatorHT::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  OutputShapeInfRetType out;
  auto host_tensor = inputs[0].toTensor();
  auto shape_tensor = inputs[1].toTensor();
  auto output = inputs[3].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana::createPTTensor(output, false);

  auto arangeOp = make_operator<ArangeOperatorHT>(
      this->p_context_->device_id_, scalar_type);
  torch::jit::Stack stack{
      IValue(host_tensor), IValue(arangeOutput), IValue(shape_tensor)};
  auto arange_op_out = out.call_ComputeOutputShape(arangeOp, stack);

  stack.clear();
  stack.emplace_back(IValue(arangeOutput));
  auto randShuffleOp = make_operator<RandomShuffleOperator>(
      this->p_context_->device_id_, scalar_type);

  auto randShuffle_op_out = out.call_ComputeOutputShape(randShuffleOp, stack);
  auto randShuffle_op_tensor = randShuffle_op_out.GetOutputTensor()[0];

  out.MoveToOutput(std::move(randShuffle_op_tensor));

  return out;
}

void RandpermOperatorHT::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
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
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input arg3 expected to be Tensor for RandpermOperatorHT");

  auto host_tensor = inputs[0].toTensor();
  auto shape_tensor = inputs[1].toTensor();
  auto output = inputs[3].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana::createPTTensor(output, false);
  auto arangeOp = make_operator<ArangeOperatorHT>(
      this->p_context_->device_id_, scalar_type);
  arangeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  arangeOp->AllocateSynapseInput(graph, arangeOutput, false);
  torch::jit::Stack stack{
      IValue(host_tensor), IValue(arangeOutput), IValue(shape_tensor)};
  arangeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // create RandomShuffle operator
  auto randShuffleOp = make_operator<RandomShuffleOperator>(
      this->p_context_->device_id_, scalar_type);
  stack.emplace_back(IValue(arangeOutput));
  randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
  randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[2]);
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
      inputs[0].isScalar() || inputs[0].isTensor(),
      "Input arg0 expected to be Scalar or Tensor for RandpermOperator operator");
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
  // {shape_tensor, seed_tensor, output_tensor}
  if (inputs[0].isTensor()) {
    auto shape_tensor = inputs[0].toTensor();
    arangeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    arangeOp->AllocateSynapseInput(graph, arangeOutput, false);
    torch::jit::Stack stack{IValue(shape_tensor), IValue(arangeOutput)};
    arangeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // create RandomShuffle operator
    auto randShuffleOp = make_operator<RandomShuffleOperator>(
        this->p_context_->device_id_, scalar_type);
    stack.emplace_back(IValue(arangeOutput));
    randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
    randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(
        std::move(randShuffleOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(randShuffleOp->GetOutputs()[0]));
  } else {
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

    // create RandomShuffle operator
    auto randShuffleOp = make_operator<RandomShuffleOperator>(
        this->p_context_->device_id_, scalar_type);
    stack.emplace_back(IValue(arangeOutput));
    randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
    randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_.emplace_back(
        std::move(randShuffleOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(randShuffleOp->GetOutputs()[0]));
  }
}

void NormalOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for Normal Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be tensor for Normal Operator");
  TORCH_CHECK(
      inputs[1].isDouble(),
      "Input arg2 expected to be Double for Normal Operator");
  TORCH_CHECK(
      inputs[2].isDouble(),
      "Input arg3 expected to be of type Double for Normal Operator");
  // For graph mode arg4 should be of type None
  TORCH_CHECK(
      inputs[3].isInt() || inputs[3].isNone(),
      "Input arg4 expected to be Int or None for Normal Operator");

  auto self = inputs[0].toTensor();
  auto mean = inputs[1].toDouble();
  auto std = inputs[2].toDouble();

  PT_KERNEL_DEBUG("mean ", mean, " ", "std ", std);

  ns_RandomNormal::Params params;
  params.mean = static_cast<float>(mean);
  params.stddev = static_cast<float>(std);

  if (inputs[3].isNone()) {
    params.seed = get_seed_hpu(c10::nullopt);
  } else {
    auto seed = inputs[3].toInt();
    params.seed = seed;
  }

  p_context_->params_.emplace<ns_RandomNormal::Params>(params);
  p_context_->params_size_ = sizeof(params);

  // Allocate Shape Tensor
  if (graph.is_dynamic_graph()) {
    AllocateSynapseShapeTensor(graph, self);
  }

  AllocateSynapseOutput(graph, self, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void BernoulliOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 2,
      "Incorrect size of inputs expected for Bernoulli Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for Bernoulli Operator");
  TORCH_CHECK(
      inputs[1].isInt() || inputs[1].isNone(),
      "Input arg2 expected to be Int or None for Bernoulli Operator");

  auto self = inputs[0].toTensor();

  ns_RandomBernoulli::Params params;

  if (inputs[1].isNone()) {
    params.seed = get_seed_hpu(c10::nullopt);
  } else {
    auto seed = inputs[1].toInt();
    params.seed = seed;
  }

  p_context_->params_.emplace<ns_RandomBernoulli::Params>(params);
  p_context_->params_size_ = sizeof(params);

  Tensor output = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void BernoulliScalarOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for BernoulliScalar Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for BernoulliScalar Operator");
  TORCH_CHECK(
      inputs[1].isDouble(),
      "Input arg2 expected to be Double for BernoulliScalar Operator");
  TORCH_CHECK(
      inputs[2].isInt() || inputs[2].isNone(),
      "Input arg3 expected to be Int or None for BernoulliScalar Operator");

  auto self = inputs[0].toTensor();
  auto p = inputs[1].toDouble();

  auto scalar_type = self.scalar_type();
  TORCH_CHECK(
      (scalar_type == c10::ScalarType::Int) ||
          (scalar_type == c10::ScalarType::Float),
      "Expected float or int data type");

  Scalar p_converted = static_cast<float>(p);

  // independent of self's dtype
  Tensor self_float = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Float,
      false);

  // Create Constant Operator to convert scalar to tensor
  auto constOp = make_operator<ConstantOperator>(
      this->p_context_->device_id_, self_float.scalar_type());
  std::vector<c10::IValue> stack = {IValue(self_float), IValue(p_converted)};
  constOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // Create Bernoulli operator
  auto brnliOp = make_operator<BernoulliOperator>(
      this->p_context_->device_id_, constOp->GetOutputs()[0].scalar_type());
  brnliOp->SetSynapseInput(constOp->GetSynOutputs()[0]);
  stack.emplace_back(IValue(constOp->GetOutputs()[0]));
  stack.emplace_back(IValue(inputs[2]));
  brnliOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  if (scalar_type == c10::ScalarType::Float) {
    // Cast Int tensor to Float tensor
    std::string node_type = "cast_i32_to_f32";

    // Create Cast operator
    auto castOp =
        make_operator<CastOperator>(this->p_context_->device_id_, node_type);
    castOp->SetSynapseInput(brnliOp->GetSynOutputs()[0]);

    stack.emplace_back(IValue(brnliOp->GetOutputs()[0]));
    stack.emplace_back(IValue(c10::ScalarType::Float));
    auto md = OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    castOp->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    // Create MemCopy operator
    auto memcopyOp = make_operator<MemCopyOperator>(
        this->p_context_->device_id_, castOp->GetOutputs()[0].scalar_type());
    memcopyOp->SetSynapseInput(castOp->GetSynOutputs()[0]);
    memcopyOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    stack.emplace_back(IValue(castOp->GetOutputs()[0]));
    stack.emplace_back(IValue(self));
    memcopyOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(memcopyOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(memcopyOp->GetOutputs()[0]));
  } else {
    // Create MemCopy operator
    auto memcopyOp = make_operator<MemCopyOperator>(
        this->p_context_->device_id_, brnliOp->GetOutputs()[0].scalar_type());
    memcopyOp->SetSynapseInput(brnliOp->GetSynOutputs()[0]);
    memcopyOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    stack.emplace_back(IValue(brnliOp->GetOutputs()[0]));
    stack.emplace_back(IValue(self));
    memcopyOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);

    p_context_->syn_outputs_.emplace_back(
        std::move(memcopyOp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(memcopyOp->GetOutputs()[0]));
  }
}

/*
Generate a seed value and push that as a tensor to HPU
*/
at::Tensor DropoutOperator::GenerateAndCopySeedToHPU(
    torch::jit::Stack& inputs,
    bool is_persistent) {
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for DropoutOperator Operator");
  // Using below approach of filling a buffer on HOST and then copying
  // to Device memory instead of doing a synMemSetD[]Async due to SW-11757
  // TODO revert to synMemSet once SW-11757 is resolved
  auto ref_tensor = inputs[0].toTensor();
  int64_t seed = inputs[2].isNone() ? get_seed_hpu(c10::nullopt)
                                    : get_seed_hpu(inputs[2].toGenerator());
  Tensor seed_tensor = habana::createPTTensor(
      ref_tensor,
      {1},
      ref_tensor.options(),
      at::MemoryFormat::Contiguous,
      c10::ScalarType::Int,
      is_persistent);
  auto size = seed_tensor.numel() * seed_tensor.element_size();
  std::vector<int> buffer(size, (int)seed);

  habana_helpers::copy_scalar_to_device(buffer.data(), seed_tensor, size);
  return seed_tensor;
}

void DropoutOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size",
      inputs.size(),
      " of inputs expected for DropoutOperator Operator");
  TORCH_CHECK(
      inputs[0].isTensor(),
      "Input arg1 expected to be Tensor for DropoutOperator Operator");
  TORCH_CHECK(
      inputs[1].isDouble(),
      "Input arg2 expected to be Double for DropoutOperator Operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg3 expected to be Tensor for DropoutOperator Operator");

  auto self = inputs[0].toTensor();
  auto p = inputs[1].toDouble();
  auto scalar_type = self.scalar_type();
  TORCH_CHECK(
      (scalar_type == c10::ScalarType::Float) ||
          (scalar_type == c10::ScalarType::BFloat16) ||
          (scalar_type == c10::ScalarType::Half),
      "Expected float, bfloat16 or half data type");

  ns_DropoutKernel::Params params;

  params.ratio = static_cast<float>(p);
  p_context_->params_.emplace<ns_DropoutKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  Tensor output = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  Tensor output_mask = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Char,
      output_metadata.at(1).persistent);
  std::vector<at::Tensor> pt_outputs{output, output_mask};
  AllocateSynapseOutputs(graph, pt_outputs, output_metadata);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

OutputShapeInfRetType DropoutOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();

  OutputShapeInfRetType out;
  // output
  out.AddOutputTensor(habana::TensorMetaData(
      self.sizes().vec(),
      HabanaOperator::CalculateStrides(
          self.sizes().vec(), self.suggest_memory_format()),
      self.scalar_type(),
      self.suggest_memory_format()));
  // output_mask
  out.AddOutputTensor(habana::TensorMetaData(
      self.sizes().vec(),
      HabanaOperator::CalculateStrides(
          self.sizes().vec(), self.suggest_memory_format()),
      c10::ScalarType::Char,
      self.suggest_memory_format()));
  return out;
}

void DropoutOperator::SetPTOutputs(
    const torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  Tensor output = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  Tensor output_mask = habana::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Char,
      output_metadata.at(1).persistent);
  std::vector<at::Tensor> pt_outputs{output, output_mask};
  HabanaOperator::SetPTOutputs(pt_outputs);
}

Tensor process_random_shuffle_op(
    const std::vector<at::Tensor>& pt_inputs,
    torch::jit::Stack& stack,
    const std::string& node_guid) {
  size_t device_id = pt_inputs[0].device().index();
  at::ScalarType scalar_type = pt_inputs[0].scalar_type();
  std::string node_type =
      node_guid + "_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  RandomShuffleOperator Op(device_id, scalar_type);

  size_t key = Op.GetRecipeKey(node_type, stack);

  if (device.get_recipe_handle_cache().isCached(key)) {
    Op.Execute(key, pt_inputs, stack);
  } else {
    // both inputs are not required, just to match graph mode stack
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");
  return out[0];
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

  auto result = graph.add_node(
      std::move(syn_inputs),
      {},
      nullptr,
      0,
      guid,
      nullptr,
      input_layouts.data(),
      nullptr,
      false);
  HABANA_ASSERT(
      ok(result),
      "Adding ",
      guid,
      " to graph failed with ",
      get_error(result).error);

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

  result = graph.add_node(
      std::move(syn_inputs),
      std::move(syn_outputs),
      nullptr,
      0,
      guid,
      nullptr,
      input_layouts.data(),
      output_layouts.data(),
      false);
  HABANA_ASSERT(
      ok(result),
      "Adding ",
      guid,
      " to graph failed with ",
      get_error(result).error);
}

static auto& RandomGenKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::randperm_out", KERNEL_FN(RandpermOperator))
        .add("hpu::randperm_out_ds", KERNEL_FN(RandpermOperator))
        .add("hpu::_fused_dropout", KERNEL_FN(DropoutOperator))
        .add("aten::_fused_dropout_backward", KERNEL_FN(DropoutOperator))
        .add("hpu::randperm_out_ds_ht", KERNEL_FN(RandpermOperatorHT))
        .add("hpu::habana_random_seed", KERNEL_FN(HabanaRandomSeedOperator));
