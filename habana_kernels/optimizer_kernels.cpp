/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <ATen/core/Reduction.h>
#include <perf_lib_layer_params.h>

#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/optimizer_kernels.h"
#include "simple_generic_kernel.h"
#include "synapse_helpers/recipe.h"

using namespace torch;
using namespace habana;

// Input tensors
// 1	Gradient             FP32/FP16/BF16	2D
// 2	Weights              FP32	2D
// 3	Moments              FP32	2D
// 4	Indices              I32	1D
// 5	Learning rate	       FP32	1D
// 6	Valid count	         I32	1D
// 7 momentum              FP32
// 8 nesterov              Bool
// Output tensors
// 1	Weights              FP32 2D
// 2	Moments              FP32	2D
#if 1 // TODO: TPC kernel seems to give wrong results.
#include "habana_helpers/graph.h"
void OptimizerSparseSgdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs expected for optimizer_sparse_sgd operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input arg6 type expected to be tensor");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isBool(), "Input arg8 type expected to be Bool");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "OptimizerSparseSgdOperator: #is_output_persistent should be 2");

  auto weights_in = inputs[1].toTensor();
  auto moments_in = inputs[2].toTensor();
  auto mom = static_cast<float>(inputs[6].toDouble());
  auto nesterov = inputs[7].toBool();

  ns_OptimizerSparseSGD::Params params;
  params.mom = mom;
  params.nesterov = nesterov;

  // execute in-place for weights & moments
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1]));
  p_context_->pt_outputs_.emplace_back(weights_in);

  // moments
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2]));
  p_context_->pt_outputs_.emplace_back(moments_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  PT_KERNEL_BEGIN;

  size_t device_id = gradients.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients.scalar_type();
  std::string node_type = "optimizer_sparse_sgd_with_valid_count_2d_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  OptimizerSparseSgdOperator Op(device_id, scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights_in),
      IValue(moments_in),
      IValue(indices),
      IValue(learning_rate),
      IValue(valid_count_tensor),
      IValue(mom),
      IValue(nesterov)};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);

    Op.SetPTInputs(pt_inputs);
    // execute in-place for weights & moments
    Op.SetPTOutputs({weights_in, moments_in});
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return std::tie(weights_in, moments_in);
}
#else
#endif

void OptimizerSparseAdagradOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 6,
      "Incorrect size of inputs expected for optimizer_adagrad_sgd operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isTensor(), "Input arg6 type expected to be tensor");
  TORCH_CHECK(
      is_output_persistent.size() == 2,
      "OptimizerSparseAdagradOperator: #is_output_persistent should be 2");

  ns_OptimizerSparseAdagrad::Params params;
  // PT does not use decay param for sparse params
  // Ref:
  // https://pytorch.org/docs/stable/_modules/torch/optim/adagrad.html#Adagrad
  // Even for dense, it applies decay param to the current grad whereas TPC
  // applies to the accumulated grad
  params.decay = 1.0;
  params.eps = 1e-10f;

  // execute in-place for weights & moments
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1]));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2]));

  auto moments_in = inputs[2].toTensor();
  p_context_->pt_outputs_.emplace_back(moments_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_KERNEL_BEGIN;

  size_t device_id = gradients.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients.scalar_type();
  std::string node_type = "optimizer_sparse_adagrad_with_valid_count_2d_" +
      habana_helpers::name_suffix_from_type(scalar_type);

  OptimizerSparseAdagradOperator Op(device_id, scalar_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{
      gradients,
      weights_in,
      moments_in,
      indices,
      learning_rate,
      valid_count_tensor};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights_in),
      IValue(moments_in),
      IValue(indices),
      IValue(learning_rate),
      IValue(valid_count_tensor)};

  size_t key = Op.GetRecipeKey(node_type, stack);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);

    Op.SetPTInputs(pt_inputs);
    // execute in-place for weights & moments
    Op.SetPTOutputs({weights_in, moments_in});
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return std::tie(weights_in, moments_in);
}

static auto& KernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habanaOptimizerSparseSgd",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerSparseSgdOperator>(
                  device_id, node_type);
            })
        .add(
            "::habanaOptimizerSparseAdagrad",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerSparseAdagradOperator>(
                  device_id, node_type);
            });