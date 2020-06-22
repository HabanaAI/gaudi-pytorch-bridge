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
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/kernel_utils.h"
#include "habana_kernels/simple_generic_kernel.h"
#include "habana_helpers/graph.h"
#include "habana_kernels/random_gen_kernels.h"

using namespace torch;

uint32_t get_seed_hpu(CPUGenerator* gen) {
  if (gen == nullptr) {
    gen = at::detail::getDefaultCPUGenerator();
  }

  // Acquire lock when using random generators
  std::lock_guard<std::mutex> lock(gen->mutex_);
  auto seed = gen->random();

  return seed;
}

void UniformOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {

  TORCH_CHECK(inputs.size() == 4, "Incorrect size of inputs expected for Uniform Operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 expected to be tensor for Uniform Operator");
  TORCH_CHECK(inputs[1].isDouble(), "Input arg2 expected to be Double for Uniform Operator");
  TORCH_CHECK(inputs[2].isDouble(), "Input arg3 expected to be of type Double for Uniform Operator");
  //For graph mode arg4 should be of type None
  TORCH_CHECK(inputs[3].isInt() || inputs[3].isNone(),
    "Input arg4 expected to be Int or None for Uniform Operator");

  auto self = inputs[0].toTensor();
  auto from = inputs[1].toDouble();
  auto to = inputs[2].toDouble();

  ns_RandomUniform::Params params;
  params.low = static_cast<float>(from);
  params.high = static_cast<float>(to);

  if(inputs[3].isNone())
  {
    params.seed = get_seed_hpu(nullptr);
  }
  else
  {
    auto seed = inputs[3].toInt();
    params.seed = seed;
  }

  p_context_->params_.emplace<ns_RandomUniform::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, self, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
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
void uniform_hpu(
    const Tensor& self,
    double from = 0,
    double to = 1,
    CPUGenerator* gen = nullptr) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "random_uniform_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  UniformOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  int64_t seed = get_seed_hpu(gen);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self),
                                    IValue(from),
                                    IValue(to),
                                    IValue(seed)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
}

void NormalOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {

  TORCH_CHECK(inputs.size() == 4, "Incorrect size of inputs expected for Normal Operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 expected to be tensor for Normal Operator");
  TORCH_CHECK(inputs[1].isDouble(), "Input arg2 expected to be Double for Normal Operator");
  TORCH_CHECK(inputs[2].isDouble(), "Input arg3 expected to be of type Double for Normal Operator");
  //For graph mode arg4 should be of type None
  TORCH_CHECK(inputs[3].isInt() || inputs[3].isNone(),
    "Input arg4 expected to be Int or None for Normal Operator");

  auto self = inputs[0].toTensor();
  auto mean = inputs[1].toDouble();
  auto std = inputs[2].toDouble();

  PT_KERNEL_DEBUG("mean ", mean, " ", "std ", std);

  ns_RandomNormal::Params params;
  params.mean = static_cast<float>(mean);
  params.stddev = static_cast<float>(std);

  if(inputs[3].isNone())
  {
    params.seed = get_seed_hpu(nullptr);
  }
  else
  {
    auto seed = inputs[3].toInt();
    params.seed = seed;
  }

  p_context_->params_.emplace<ns_RandomNormal::Params>(params);
  p_context_->params_size_ = sizeof(params);

  AllocateSynapseOutput(graph, self, is_output_persistent);
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
void normal_hpu(
    const Tensor& self,
    double mean = 0,
    double std = 1,
    CPUGenerator* gen = nullptr) {
  PT_KERNEL_BEGIN;

  at::ScalarType scalar_type = self.scalar_type();
  std::string node_type =
      "random_normal_fwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  size_t device_id = self.device().index();

  NormalOperator Op(device_id, node_type);
  // Create Graph
  auto graph = habana_helpers::create_graph(device_id, node_type);

  int64_t seed = get_seed_hpu(gen);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(self),
                                    IValue(mean),
                                    IValue(std),
                                    IValue(seed)};
  Op.AllocateAndAddSynapseNode(graph, stack, true);

  // compile and execute the graph
  Op.Compile(graph);

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
}

/*******************************************************************
*@brief Implements Bernoulli distribution generation kernel
@param[in] self - output tensor with probablities, 1-4D,
BF16/FP32
@param[in] gen - Generator class for seed (optional)
@param[out] - Tensor same shape as self with 0/1 entries generated based on
input probabilities , I16/I32, 1-4D
*******************************************************************/
Tensor bernoulli_hpu(const Tensor& self, CPUGenerator* gen = nullptr) {
  PT_KERNEL_BEGIN;

  Tensor output =
      at::empty(self.sizes(), self.options().dtype(c10::ScalarType::Int));

  std::vector<const at::Tensor*> pt_inputs{&self};
  std::vector<const at::Tensor*> pt_outputs{&output};

  ns_RandomBernoulli::Params params;
  params.seed = get_seed_hpu(gen);

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "random_bernoulli",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  PT_KERNEL_END;

  return output;
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
    CPUGenerator* gen = nullptr) {
  PT_KERNEL_BEGIN;

  auto self_scalar_type = self.scalar_type();

  TORCH_CHECK(
      (self_scalar_type == c10::ScalarType::Int) ||
          (self_scalar_type == c10::ScalarType::Float),
      "Expected float or int data type");

  Scalar p_converted = static_cast<float>(p);

  auto p_tensor = habana_helpers::scalar_to_device_tensor(
      p_converted,
      self.options().dtype(c10::ScalarType::Float),
      self.ndimension());
  auto expanded_p_tensor = p_tensor.expand(self.sizes());

  Tensor* output_ptr;
  Tensor self_int;

  std::vector<const at::Tensor*> pt_inputs{&expanded_p_tensor};

  if (self_scalar_type == c10::ScalarType::Float) {
    self_int =
        at::empty(self.sizes(), self.options().dtype(c10::ScalarType::Int));
    output_ptr = &self_int;
  } else {
    // Int
    output_ptr = &self;
  }

  std::vector<const at::Tensor*> pt_outputs{output_ptr};
  ns_RandomBernoulli::Params params;
  params.seed = get_seed_hpu(gen);

  synapse_simple_generic_kernel(
      pt_outputs,
      pt_inputs,
      "random_bernoulli",
      &params,
      sizeof(params),
      SynapsePassType::FORWARD_PASS);

  if (self_scalar_type == c10::ScalarType::Float) {
    auto self_float = habana_helpers::hpu_cast_tensor(self_int, self.dtype());
    habana_helpers::copy_data_within_device(self_float, self);
  }

  PT_KERNEL_END;

  return self;
}

static auto registry =
    torch::RegisterOperators()
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::uniform_(Tensor(a!) self, float from=0, float to=1, *, Generator? generator=None) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(uniform_hpu), &uniform_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::normal_(Tensor(a!) self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<decltype(normal_hpu), &normal_hpu>(
                    DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor")
                .impl_unboxedOnlyKernel<
                    decltype(bernoulli_hpu),
                    &bernoulli_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA))
        .op(torch::RegisterOperators::options()
                .schema(
                    "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)")
                .impl_unboxedOnlyKernel<
                    decltype(bernoulli_scalar_hpu),
                    &bernoulli_scalar_hpu>(DispatchKey::HABANATensorId)
                .aliasAnalysis(c10::AliasAnalysisKind::FROM_SCHEMA));
