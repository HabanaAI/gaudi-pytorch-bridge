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
// 1	Weights              FP32/FP16/BF16	2D
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

  auto weights_out =
      habana_helpers::createPTTensor(weights_in, is_output_persistent[0]);
  auto moments_out =
      habana_helpers::createPTTensor(moments_in, is_output_persistent[1]);

  AllocateSynapseOutputs(
      graph, {weights_out, moments_out}, is_output_persistent);
  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

std::tuple<torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_hpu(
    const Tensor& gradients,
    const Tensor& weights_in,
    const Tensor& moments_in,
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

  OptimizerSparseSgdOperator Op(device_id, node_type);
  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs{gradients,
                                    weights_in,
                                    moments_in,
                                    indices,
                                    learning_rate,
                                    valid_count_tensor};
  // Build Params for the graph
  std::vector<c10::IValue> stack = {IValue(gradients),
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
    auto weights_out = habana_helpers::createPTTensor(weights_in, true);
    auto moments_out = habana_helpers::createPTTensor(moments_in, true);

    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs({weights_out, moments_out});
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true, true});
    // compile and execute the graph
    Op.Compile(graph);
  }
  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 2, "Incorrect size of outputs");

  PT_KERNEL_END;
  return std::tie(out.at(0), out.at(1));
}
#else
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_cpu(
    torch::Tensor gradients,
    torch::Tensor weights_in,
    torch::Tensor moments_in,
    torch::Tensor indices,
    torch::Tensor learning_rate,
    int64_t valid_count,
    float mom,
    bool nesterov) {
  /*
  moments_out[sparse_indices] = momentum_in[sparse_indices] * state.mom +
                                gradients[sparse_indices];
  gradients_out[sparse_indices] = momentum_out[sparse_indices];
  weights_out[sparse_indices] =
    weights_in[sparse_indices] - state.lr * gradients_out[sparse_indices];
  */
  auto sizes = weights_in.sizes().vec();
  float* gp = static_cast<float*>(gradients.data_ptr());
  float* winp = static_cast<float*>(weights_in.data_ptr());
  float* minp = static_cast<float*>(moments_in.data_ptr());
  Tensor weights_out = at::empty(
      weights_in.sizes(),
      weights_in.options(),
      weights_in.suggest_memory_format());
  Tensor moments_out = at::empty(
      moments_in.sizes(),
      moments_in.options(),
      moments_in.suggest_memory_format());
  Tensor grad_output = at::empty(
      weights_in.sizes(),
      gradients.options(),
      gradients.suggest_memory_format());
  weights_out.copy_(weights_in, false);
  moments_out.copy_(moments_in, false);
  grad_output.copy_(weights_in, false);
  float* woutp = static_cast<float*>(weights_out.data_ptr());
  float* moutp = static_cast<float*>(moments_out.data_ptr());
  float* goutp = static_cast<float*>(grad_output.data_ptr());
  int* inp = static_cast<int*>(indices.data_ptr());
  float* lrp = static_cast<float*>(learning_rate.data_ptr());
  float gtemp;
  unsigned vec_len = sizes[1];
  for (unsigned i = 0; i < valid_count; i++) {
    for (unsigned k = 0; k < vec_len; k++) {
      // momentum update
      moutp[inp[i] * vec_len + k] =
          minp[inp[i] * vec_len + k] * mom + gp[i * vec_len + k];
      gtemp = moutp[inp[i] * vec_len + k];
      // grad update
      if (nesterov) {
        goutp[inp[i] * vec_len + k] = gp[inp[i] * vec_len + k] + mom * gtemp;
      } else {
        goutp[inp[i] * vec_len + k] = gtemp;
      }
      // weight update
      woutp[inp[i] * vec_len + k] = winp[inp[i] * vec_len + k] - *lrp * gtemp;
    }
  }
  return std::make_tuple(weights_out, moments_out, grad_output);
}

std::tuple<torch::Tensor, torch::Tensor>
optimizer_sparse_sgd_with_valid_count_hpu(
    const torch::Tensor& gradients,
    const torch::Tensor& weights_in,
    const torch::Tensor& moments_in,
    const torch::Tensor& indices,
    const torch::Tensor& learning_rate,
    int64_t valid_count,
    float mom,
    bool nesterov) {
  PT_KERNEL_BEGIN;
  auto sizes = weights_in.sizes().vec();
  for (unsigned int i = 0; i < weights_in.dim(); i++)
    PT_KERNEL_DEBUG("sizes = ", sizes[i]);
  auto cast_indices = habana_helpers::cast_tensor_to_integer(indices);
  auto hpu = indices.device();
  auto result = optimizer_sparse_sgd_with_valid_count_cpu(
      gradients.to("cpu"),
      weights_in.to("cpu"),
      moments_in.to("cpu"),
      cast_indices.to("cpu"),
      learning_rate.to("cpu"),
      valid_count,
      mom,
      nesterov);
  auto ret1 = std::get<0>(result);
  auto ret2 = std::get<1>(result);
  PT_KERNEL_END;
  return std::make_tuple(ret1.to(hpu), ret2.to(hpu));
}

#endif
