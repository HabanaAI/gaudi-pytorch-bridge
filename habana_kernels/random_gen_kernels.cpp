/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <ATen/ExpandUtils.h>
#include <perf_lib_layer_params.h>
#include <torch/script.h>
#include <memory>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/basic_kernels.h"
#include "habana_kernels/index_kernels.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_kernels/resize.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "lazy_kernels.h"

using namespace torch;

namespace habana {
uint32_t get_seed_hpu(const c10::optional<Generator>& gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(
      gen, at::detail::getDefaultCPUGenerator());

  // Acquire lock when using random generators
  std::lock_guard<std::mutex> lock(generator->mutex_);
  return generator->random();
}

at::Tensor get_seed_tensor_hpu(const c10::optional<Generator>& gen) {
  int seed = get_seed_hpu(gen);
  at::Tensor seed_tensor = at::tensor(seed);
  return habana_lazy::append_to_batch_h2d_list(seed_tensor);
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

at::Tensor RandpermOperator::GenerateAndCopySeedToHPU(
    torch::jit::Stack& inputs,
    bool is_persistent) {
  auto generate_seed = [&](IValue seed_val, IValue tensor_val) {
    // Using below approach of filling a buffer on HOST and then copying
    // to Device memory instead of doing a synMemSetD[]Async due to SW-11757
    // TODO revert to synMemSet once SW-11757 is resolved
    auto ref_tensor = tensor_val.toTensor();
    int64_t seed = seed_val.isNone() ? get_seed_hpu(c10::nullopt)
                                     : get_seed_hpu(seed_val.toGenerator());
    Tensor seed_tensor = habana_helpers::createPTTensor(
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
  };
  if (inputs.size() == 3) {
    TORCH_CHECK(
        inputs[2].isTensor(),
        "Input arg1 expected to be Tensor for RandpermOperator Operator");
    return generate_seed(inputs[1], inputs[2]);
  } else {
    TORCH_CHECK(
        inputs.size() == 4, "GenerateAndCopySeedToHPU: input size incorrect");
    TORCH_CHECK(
        inputs[3].isTensor(),
        "Input arg1 expected to be Tensor for RandpermOperatorHT Operator")
    return generate_seed(inputs[2], inputs[3]);
  }
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
      "Input arg0 expected to be Tensor for RandpermOperatorHT");
  TORCH_CHECK(
      inputs[2].isGenerator() || inputs[2].isNone(),
      "Input arg1 expected to be Generator for RandpermOperatorHT");
  TORCH_CHECK(
      inputs[3].isTensor(),
      "Input arg2 expected to be Tensor for RandpermOperatorHT");

  auto host_tensor = inputs[0].toTensor();
  auto shape_tensor = inputs[1].toTensor();
  auto output = inputs[3].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana_helpers::createPTTensor(output, false);
  auto arangeOp = make_operator<ArangeOperatorHT>(
      this->p_context_->device_id_, scalar_type);
  arangeOp->SetSynapseInput(p_context_->syn_inputs_[0]);
  arangeOp->AllocateSynapseInput(graph, arangeOutput, false);
  torch::jit::Stack stack{
      IValue(host_tensor), IValue(arangeOutput), IValue(shape_tensor)};
  arangeOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();
  // Move inputs[2] as output tensor
  synapse_helpers::tensor& syn_out_t = p_context_->syn_inputs_[2];
  p_context_->syn_outputs_.emplace_back(syn_out_t);
  p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[2]);
  p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin() + 2);
  p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin() + 2);

  // create RandomShuffle operator
  auto randShuffleOp = make_operator<RandomShuffleOperator>(
      this->p_context_->device_id_, scalar_type);
  stack.emplace_back(IValue(arangeOutput));
  randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
  randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[2]);
  randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  p_context_->syn_outputs_[0] = std::move(randShuffleOp->GetSynOutputs()[0]);
  p_context_->pt_outputs_[0] = std::move(randShuffleOp->GetOutputs()[0]);
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
      inputs[1].isGenerator() || inputs[1].isNone(),
      "Input arg1 expected to be Generator for RandpermOperator Operator");
  TORCH_CHECK(
      inputs[2].isTensor(),
      "Input arg2 expected to be Tensor for RandpermOperator Operator");

  auto output = inputs[2].toTensor();
  auto scalar_type = output.scalar_type();
  auto arangeOutput = habana_helpers::createPTTensor(output, false);
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
    // Move inputs[1] as output tensor
    synapse_helpers::tensor& syn_out_t = p_context_->syn_inputs_[1];
    p_context_->syn_outputs_.emplace_back(syn_out_t);
    p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[1]);
    p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin() + 1);
    p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin() + 1);

    // create RandomShuffle operator
    auto randShuffleOp = make_operator<RandomShuffleOperator>(
        this->p_context_->device_id_, scalar_type);
    stack.emplace_back(IValue(arangeOutput));
    randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
    randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[1]);
    randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_[0] = std::move(randShuffleOp->GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(randShuffleOp->GetOutputs()[0]);
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
    // Move inputs[0] as output tensor
    synapse_helpers::tensor& syn_out_t = p_context_->syn_inputs_[0];
    p_context_->syn_outputs_.emplace_back(syn_out_t);
    p_context_->pt_outputs_.emplace_back(p_context_->pt_inputs_[0]);
    p_context_->syn_inputs_.erase(p_context_->syn_inputs_.begin());
    p_context_->pt_inputs_.erase(p_context_->pt_inputs_.begin());

    // create RandomShuffle operator
    auto randShuffleOp = make_operator<RandomShuffleOperator>(
        this->p_context_->device_id_, scalar_type);
    stack.emplace_back(IValue(arangeOutput));
    randShuffleOp->SetSynapseInput(arangeOp->GetSynOutputs()[0]);
    randShuffleOp->SetSynapseInput(p_context_->syn_inputs_[0]);
    randShuffleOp->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    p_context_->syn_outputs_[0] = std::move(randShuffleOp->GetSynOutputs()[0]);
    p_context_->pt_outputs_[0] = std::move(randShuffleOp->GetOutputs()[0]);
  }
}

/*******************************************************************
*@brief Implements uniform distribution generation kernel
*INPUTS
@param[in, out] self - output tensor with uniform distributed values, 2D/3D/4D,
bf16/FP32
@param[in] from - lower bound
@param[in] to - upper bound
@param[in] gen - Generator class for seed (optional)
*******************************************************************/
Tensor& uniform_hpu(
    Tensor& self,
    double from = 0,
    double to = 1,
    c10::optional<Generator> gen = c10::nullopt) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "random_uniform_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  UniformOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  int64_t seed = get_seed_hpu(gen);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(from), IValue(to), IValue(seed)};
  OutputMetaDataVector output_metadata(1);
  output_metadata.at(0).persistent = true;
  Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return self;
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

/*******************************************************************
*@brief Implements normal distribution generation kernel
*INPUTS
@param[in, out] self - output tensor with normal distributed values, 2D/3D/4D,
bf16/FP32
@param[in] mean, default = 0
@param[in] std, default = 1
@param[in] gen - Generator class for seed (optional)
*******************************************************************/
Tensor& normal_hpu(
    Tensor& self,
    double mean = 0,
    double std = 1,
    c10::optional<Generator> gen = c10::nullopt) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "random_normal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  NormalOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  int64_t seed = get_seed_hpu(gen);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(mean), IValue(std), IValue(seed)};
  OutputMetaDataVector output_metadata(1);
  output_metadata.at(0).persistent = true;
  Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return self; // out.at(0);
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

  Tensor output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Int,
      output_metadata.at(0).persistent);
  AllocateSynapseOutput(graph, output, output_metadata.at(0));
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}
/*******************************************************************
*@brief Implements Bernoulli distribution generation kernel
@param[in] self - output tensor with probablities, 1-4D,
BF16/FP32
@param[in] gen - Generator class for seed (optional)
@param[out] - Tensor same shape as self with 0/1 entries generated based on
input probabilities , I16/I32, 1-4D
*******************************************************************/
Tensor bernoulli_hpu(const Tensor& self, c10::optional<Generator> gen) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "random_bernoulli_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  BernoulliOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  int64_t seed = get_seed_hpu(gen);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(seed)};
  OutputMetaDataVector output_metadata(1);
  output_metadata.at(0).persistent = true;
  Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out.at(0);
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
  Tensor self_float = habana_helpers::createPTTensor(
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
    castOp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
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

/*******************************************************************
*@brief Implements Bernoulli distribution generation kernel
*INPUTS self.bernoulli_(p=0.5, *, generator=None) → Tensor
*Fills each location of self with an independent sample from
Bernoulli(p).
@param[in, out] self - output tensor with probablities, 1-4D,
BF16/FP32.
@param[in] gen - Generator class for seed (optional)
*******************************************************************/
Tensor& bernoulli_scalar_hpu(
    Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type = "random_bernoulli_fwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  BernoulliScalarOperator Op(device_id, scalar_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  std::vector<at::Tensor> pt_inputs{self};
  Op.AllocateSynapseInputs(graph, pt_inputs, true);

  int64_t seed = get_seed_hpu(gen);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self), IValue(p), IValue(seed)};
  OutputMetaDataVector output_metadata(1);
  output_metadata.at(0).persistent = true;
  Op.AllocateAndAddSynapseNode(graph, stack, output_metadata);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return self;
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
  Tensor seed_tensor = habana_helpers::createPTTensor(
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
      (scalar_type == c10::ScalarType::BFloat16) ||
          (scalar_type == c10::ScalarType::Float),
      "Expected float or int data type");

  ns_DropoutKernel::Params params;

  params.ratio = static_cast<float>(p);
  p_context_->params_.emplace<ns_DropoutKernel::Params>(params);
  p_context_->params_size_ = sizeof(params);

  Tensor output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  Tensor output_mask = habana_helpers::createPTTensor(
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

void DropoutOperator::SetPTOutputs(
    const torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  auto self = inputs[0].toTensor();
  Tensor output = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      self.scalar_type(),
      output_metadata.at(0).persistent);
  Tensor output_mask = habana_helpers::createPTTensor(
      self,
      self.sizes(),
      self.options(),
      self.suggest_memory_format(),
      c10::ScalarType::Char,
      output_metadata.at(1).persistent);
  std::vector<at::Tensor> pt_outputs{output, output_mask};
  HabanaOperator::SetPTOutputs(pt_outputs);
}

/*******************************************************************
*@brief Implements Dropout kernel
@param[in] self - input tensor on which Dropout is applied
@param[in] p - probability of dropped out connections
@param[in] gen - Generator class for seed (optional)
*This function is probably not going to be called as the
*aten::_fused_dropout() kernel should be called only in graph mode
*******************************************************************/
std::tuple<Tensor, Tensor> fused_dropout_hpu(
    const Tensor& self,
    double p,
    c10::optional<Generator> gen) {
  PT_KERNEL_BEGIN;
  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "dropout_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  DropoutOperator Op(device_id, scalar_type);

  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);
  // Build Params for the graph
  auto seed_tensor = habana::get_seed_tensor_hpu(gen);
  std::vector<c10::IValue> stack = {
      IValue(self), IValue(p), IValue(seed_tensor)};
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{self, seed_tensor};
  size_t key = Op.GetRecipeKey(node_type, stack);
  OutputMetaDataVector output_metadata(2);
  output_metadata.at(0).persistent = true;
  output_metadata.at(1).persistent = true;

  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(stack, output_metadata);
    Op.Execute(key);
  } else {
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::tuple<Tensor, Tensor>(out.at(0), out.at(1));
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

Tensor random_shuffle_tensor_hpu(const Tensor& self, const Tensor& seed) {
  PT_KERNEL_BEGIN;

  auto self_hpu = self.to(c10::DeviceType::HPU);
  auto seed_hpu = seed.to(c10::DeviceType::HPU);

  std::vector<at::Tensor> pt_inputs{self_hpu, seed_hpu};
  torch::jit::Stack stack{IValue(self_hpu)};

  auto output = process_random_shuffle_op(pt_inputs, stack, "random_shuffle");

  PT_KERNEL_END;
  return output;
}

Tensor& randperm_hpu(Tensor& output, int64_t n, c10::optional<Generator> gen) {
  PT_KERNEL_BEGIN;

  auto shape = DimVector({n});
  Scalar n_scalar((int32_t(n)));
  auto tht_result = output.unsafeGetTensorImpl();
  THHTensor_resizeNd(tht_result, shape.size(), shape.data(), nullptr);

  auto scalar_type = output.scalar_type();

  std::string node_type =
      "randperm_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = output.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  RandpermOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<at::Tensor> pt_inputs{output};
  std::vector<at::Tensor> pt_outputs{output};
  std::vector<c10::IValue> stack = {
      IValue(n_scalar), IValue(gen), IValue(output)};

  // create seed tensor
  auto seed_tensor = RandpermOperator::GenerateAndCopySeedToHPU(stack, true);
  pt_inputs.emplace_back(seed_tensor);

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    pt_inputs.erase(pt_inputs.begin());
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(pt_outputs);
    Op.Execute(key, pt_inputs, pt_outputs);
  } else {
    OutputMetaDataVector output_metadata(1);
    output_metadata.at(0).persistent = true;
    // compile and execute the graph
    Op.CreateGraphAndCompile(key, pt_inputs, stack, output_metadata, true);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  output.copy_(out.at(0));
  PT_KERNEL_END;

  return output;
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add("hpu::randperm_out", KERNEL_FN(RandpermOperator))
        .add("hpu::randperm_out_ds", KERNEL_FN(RandpermOperator))
        .add("hpu::_fused_dropout", KERNEL_FN(DropoutOperator))
        .add("aten::_fused_dropout_backward", KERNEL_FN(DropoutOperator))
        .add("hpu::randperm_out_ds_ht", KERNEL_FN(RandpermOperatorHT));
