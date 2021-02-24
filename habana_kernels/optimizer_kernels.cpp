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
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/optimizer_kernels.h"
#include "habana_kernels/unary_kernels.h"
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
      "Incorrect size of inputs for optimizer_sparse_sgd operator");
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
      "Incorrect size of inputs for optimizer_adagrad_sgd operator");
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

/*
Generate tensors for LR and neg_step and copy to HPU
*/
std::tuple<Tensor, Tensor> OptimizerAdamwOperator::GenerateAndCopyTensorsToHPU(
    const Tensor& ref_tensor,
    const float lr,
    const float neg_step,
    bool is_persistent) {
  // Convert neg_step and lr to tensors to avoid cache misses
  Tensor neg_step_t = habana_helpers::createPTTensor(
      ref_tensor,
      {1},
      ref_tensor.options(),
      ref_tensor.suggest_memory_format(),
      c10::ScalarType::Float,
      is_persistent);
  auto size1 = neg_step_t.numel() * neg_step_t.element_size();
  std::vector<float> buffer1(size1, neg_step);
  habana_helpers::copy_scalar_to_device(buffer1.data(), neg_step_t, size1);

  Tensor lr_t = habana_helpers::createPTTensor(
      ref_tensor,
      {1},
      ref_tensor.options(),
      ref_tensor.suggest_memory_format(),
      c10::ScalarType::Float,
      is_persistent);
  auto size2 = lr_t.numel() * lr_t.element_size();
  std::vector<float> buffer2(size2, lr);
  habana_helpers::copy_scalar_to_device(buffer2.data(), lr_t, size2);

  return std::tuple<Tensor, Tensor>(neg_step_t, lr_t);
}

void OptimizerAdamwOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 10,
      "Incorrect size of inputs for adamw optimizer graph creation call");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto exp_avg = inputs[2].toTensorList();
  auto exp_avg_sq = inputs[3].toTensorList();
  UNUSED auto lr = inputs[4].toTensor();
  auto beta1 = inputs[5].toScalar();
  auto beta2 = inputs[6].toScalar();
  auto epsilon = inputs[7].toScalar();
  auto neg_step_size = inputs[8].toTensor();
  UNUSED auto weight_decay = inputs[9].toScalar();

  /*  This are the operations we need to perform per parameter
      exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
      denom = exp_avg_sq.sqrt().add_(group["eps"])
      ratio = torch.div(exp_avg, denom)
      scaled_ratio = torch.mul(ratio, step_size)
      p.data.sub_(scaled_ratio)
      if group["weight_decay"] > 0.0:
        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
  */
  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();
  auto num_params = static_cast<int>(weights.size());
  torch::jit::Stack stack;
  for (auto i = 0; i < num_params; i++) {
    // Synapse Graph for single parameter update to be created here
    // All synapse input tensor references are there in a single std::vector
    // gradients ; weights ; exp_avg ; exp_avg_sq ; lr ; neg_step_size

    // exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    habana::MulInplaceOperator mul_exp_avg(device_id, scalar_type);
    auto& syn_in_10 = mul_exp_avg.SetSynapseInput(
        std::move(p_context_->syn_inputs_[2 * num_params + i]));
    stack.emplace_back(IValue(exp_avg.get(i)));
    stack.emplace_back(IValue(beta1));
    mul_exp_avg.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[2 * num_params + i] = std::move(syn_in_10);
    stack.clear();

    habana::AddInplaceOperator add_exp_avg(device_id, scalar_type);
    auto& syn_in_11 =
        add_exp_avg.SetSynapseInput(std::move(mul_exp_avg.GetSynOutputs()[0]));
    auto& syn_in_21 =
        add_exp_avg.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    stack.emplace_back(IValue(mul_exp_avg.GetOutputs()[0]));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(Scalar(1.0 - beta1.toDouble())));
    add_exp_avg.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[i] = std::move(syn_in_21);
    stack.clear();

    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    habana::MulInplaceOperator mul_exp_avg_sq(device_id, scalar_type);
    auto& syn_in_13 = mul_exp_avg_sq.SetSynapseInput(
        std::move(p_context_->syn_inputs_[3 * num_params + i]));
    stack.emplace_back(IValue(exp_avg_sq.get(i)));
    stack.emplace_back(IValue(beta2));
    mul_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[3 * num_params + i] = std::move(syn_in_13);
    stack.clear();

    habana::AddcmulInplaceOperator addcmul_exp_avg_sq(device_id, scalar_type);
    auto& syn_in_14 = addcmul_exp_avg_sq.SetSynapseInput(
        std::move(mul_exp_avg_sq.GetSynOutputs()[0]));
    auto& syn_in_24 = addcmul_exp_avg_sq.SetSynapseInput(
        std::move(p_context_->syn_inputs_[i]));
    // Internally we are going to use "pow" instead of "mul",
    // therefore 3rd synapse tensor will be unused. We can give
    // a dummy tensor
    UNUSED auto& syn_in_3 = addcmul_exp_avg_sq.SetSynapseInput(
        std::move(habana_helpers::create_tensor(
            gradients.get(i), graph.get_graph_handle(), true, c10::nullopt)));
    stack.emplace_back(IValue(mul_exp_avg_sq.GetOutputs()[0]));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(Scalar(1.0 - beta2.toDouble())));
    addcmul_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[i] = std::move(syn_in_24);
    stack.clear();

    // denom = exp_avg_sq.sqrt().add_(group["eps"])
    // we will actually do "add" instead of "add_". Inplace not strictly
    // required here
    SqrtOperator sqrt_exp_avg_sq(device_id, scalar_type);
    auto& syn_in_15 = sqrt_exp_avg_sq.SetSynapseInput(
        std::move(addcmul_exp_avg_sq.GetSynOutputs()[0]));
    stack.emplace_back(IValue(addcmul_exp_avg_sq.GetOutputs()[0]));
    sqrt_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    habana::AddOperator add_exp_avg_sq(device_id, scalar_type);
    UNUSED auto& syn_in_16 = add_exp_avg_sq.SetSynapseInput(
        std::move(sqrt_exp_avg_sq.GetSynOutputs()[0]));
    stack.emplace_back(IValue(sqrt_exp_avg_sq.GetOutputs()[0]));
    stack.emplace_back(IValue(epsilon));
    stack.emplace_back(IValue(1.0));
    add_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Replaced addcdiv with following OPs, so that -step_size
    // can be used as a tensor
    // ratio = torch.div(exp_avg, denom)
    // scaled_ratio = torch.mul(ratio, -step_size)
    // p.data.add_(scaled_ratio)
    habana::DivOperator div_wt(device_id, scalar_type);
    auto& syn_in_17 =
        div_wt.SetSynapseInput(std::move(add_exp_avg.GetSynOutputs()[0]));
    UNUSED auto& syn_in_27 =
        div_wt.SetSynapseInput(std::move(add_exp_avg_sq.GetSynOutputs()[0]));
    stack.emplace_back(IValue(add_exp_avg.GetOutputs()[0]));
    stack.emplace_back(IValue(add_exp_avg_sq.GetOutputs()[0]));
    div_wt.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    habana::MulOperator mul_wt(device_id, scalar_type);
    UNUSED auto& syn_in_18 =
        mul_wt.SetSynapseInput(std::move(div_wt.GetSynOutputs()[0]));
    auto& syn_in_28 = mul_wt.SetSynapseInput(
        std::move(p_context_->syn_inputs_[4 * num_params + 1]));
    stack.emplace_back(IValue(div_wt.GetOutputs()[0]));
    stack.emplace_back(IValue(neg_step_size));
    mul_wt.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[4 * num_params + 1] = std::move(syn_in_28);
    stack.clear();

    habana::AddInplaceOperator add_wt(device_id, scalar_type);
    auto& syn_in_19 = add_wt.SetSynapseInput(
        std::move(p_context_->syn_inputs_[1 * num_params + i]));
    UNUSED auto& syn_in_29 =
        add_wt.SetSynapseInput(std::move(mul_wt.GetSynOutputs()[0]));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(mul_wt.GetOutputs()[0]));
    stack.emplace_back(IValue(1.0));
    add_wt.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[1 * num_params + i] = std::move(syn_in_19);
    stack.clear();

    // if group["weight_decay"] > 0.0:
    //  p.data.add_(p.data, alpha=-group["lr"] *
    //  group["weight_decay"])
    // Not going to implement this in 1st pass. Its not being used in
    // BERT-L Hugging-face scripts

    // Note that these outputs are being filled just to keep GC
    // runtime happy No need to return these since updates on
    // weights, exp_avg, exp_avg_sq are all inplace
    p_context_->syn_outputs_.emplace_back(std::move(syn_in_11));
    p_context_->pt_outputs_.emplace_back(mul_exp_avg.GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(std::move(syn_in_17));
    p_context_->pt_outputs_.emplace_back(add_exp_avg.GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(std::move(syn_in_14));
    p_context_->pt_outputs_.emplace_back(mul_exp_avg_sq.GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(std::move(syn_in_15));
    p_context_->pt_outputs_.emplace_back(addcmul_exp_avg_sq.GetOutputs()[0]);
    p_context_->syn_outputs_.emplace_back(std::move(add_wt.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(add_wt.GetOutputs()[0]);
  }
}

void optimizer_adamw_hpu(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  PT_KERNEL_BEGIN;

  TensorList gradients(gradient_vec);
  TensorList weights(weight_vec);
  TensorList exp_avg(exp_avg_vec);
  TensorList exp_avg_sq(exp_avg_sq_vec);

  std::vector<c10::IValue> orig_stack = {
      IValue(gradients),
      IValue(weights),
      IValue(exp_avg),
      IValue(exp_avg_sq),
      IValue(lr),
      IValue(beta1),
      IValue(beta2),
      IValue(epsilon),
      IValue(step),
      IValue(bias_correction),
      IValue(weight_decay)};
  auto step_size = lr;
  if (bias_correction) {
    auto bias_correction1 = 1.0 - std::pow(beta1, step);
    auto bias_correction2 = 1.0 - std::pow(beta2, step);
    step_size = step_size * std::sqrt(bias_correction2) / bias_correction1;
  }
  // Negate step here itself before converting it to tensor
  // easier then negating after conversion to tensor
  auto neg_step = -step_size;
  auto ret_tuple = OptimizerAdamwOperator::GenerateAndCopyTensorsToHPU(
      gradients[0], lr, neg_step, true);
  auto neg_step_t = std::get<0>(ret_tuple);
  auto lr_t = std::get<1>(ret_tuple);

  size_t device_id = gradients[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients[0].scalar_type();
  std::string node_type =
      "optimizer_adamw_" + habana_helpers::name_suffix_from_type(scalar_type);
  OptimizerAdamwOperator Op(device_id, scalar_type);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights),
      IValue(exp_avg),
      IValue(exp_avg_sq),
      IValue(lr_t),
      IValue(beta1),
      IValue(beta2),
      IValue(epsilon),
      IValue(neg_step_t),
      IValue(weight_decay)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs;
  std::vector<at::Tensor> pt_outputs;
  auto num_params = static_cast<int>(gradients.size());
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(gradients[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(weights[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(exp_avg[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(exp_avg_sq[j]);
  }
  pt_inputs.push_back(lr_t);
  pt_inputs.push_back(neg_step_t);

  for (auto j = 0; j < num_params; j++) {
    pt_outputs.push_back(exp_avg[j]);
    pt_outputs.push_back(exp_avg[j]);
    pt_outputs.push_back(exp_avg_sq[j]);
    pt_outputs.push_back(exp_avg_sq[j]);
    pt_outputs.push_back(weights[j]);
  }
  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutputs(pt_outputs);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return;
}

void OptimizerAdagradOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs for optimizer_adagrad operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");

  auto gradients = inputs[0].toTensor();
  auto weights = inputs[1].toTensor();
  auto variances = inputs[2].toTensor();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();

  // std::cout << "weight size "
  //           << weights.sizes() << std::endl;

  ns_OptimizerAdagrad::Params params;
  params.wd = inputs[5].toDouble();
  params.lrd = inputs[6].toDouble();
  params.eps = inputs[7].toDouble();

  // execute in-place for weights & variance
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1]));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2]));

  auto variance_in = inputs[2].toTensor();
  p_context_->pt_outputs_.emplace_back(variance_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
}

void OptimizerFusedAdagradOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs for optimizer fused adagrad operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[2].isTensorList(), "Input arg3 type expected to be tensorlist");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto variances = inputs[2].toTensorList();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();

  auto num_params = static_cast<int>(gradients.size());

  torch::jit::Stack stack;
  size_t device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  for (auto i = 0; i < num_params; i++) {
    OptimizerAdagradOperator op(device_id, scalar_type);
    auto& syn_grad = op.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    auto& syn_wt =
        op.SetSynapseInput(std::move(p_context_->syn_inputs_[num_params + i]));
    auto& syn_var = op.SetSynapseInput(
        std::move(p_context_->syn_inputs_[2 * num_params + i]));
    auto& syn_epoch_num =
        op.SetSynapseInput(std::move(p_context_->syn_inputs_[3 * num_params]));
    auto& syn_lr = op.SetSynapseInput(
        std::move(p_context_->syn_inputs_[3 * num_params + 1]));

    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(variances.get(i)));
    stack.emplace_back(inputs[3]);
    stack.emplace_back(inputs[4]);
    stack.emplace_back(inputs[5]);
    stack.emplace_back(inputs[6]);
    stack.emplace_back(inputs[7]);

    op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    stack.clear();

    p_context_->syn_inputs_[i] = std::move(syn_grad);
    p_context_->syn_inputs_[num_params + i] = std::move(syn_wt);
    p_context_->syn_inputs_[2 * num_params + i] = std::move(syn_var);
    p_context_->syn_inputs_[3 * num_params] = std::move(syn_epoch_num);
    p_context_->syn_inputs_[3 * num_params + 1] = std::move(syn_lr);

    p_context_->syn_outputs_.emplace_back(std::move(op.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(op.GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(op.GetSynOutputs()[1]));
    p_context_->pt_outputs_.emplace_back(op.GetOutputs()[1]);

  } // for (auto i = 0;i < num_params;i++)
}

/*************************************************************************************
@brief - Implements custom fused adgrad optimizer for dense parameters
@param[in] - gradients - TensorList of gradients tensors FP32, 2D
@param[in, out] - weights - TensorList of gradients tensors FP32, 2D
@param[in, out] - variances - TensorList of weight variance FP32, 2D
@param[in] - epoch_num - current epoch training number - I32, 1D
@param[in] - learning rate - FP32, 1D
@param[in] - wd - weight decay - FP32
@param[in] - lrd - learning rate decay - FP32
@param[in] - epsilon - constant to avoid division by zero, FP32

@param[out] - lr - This is dummy output to be compliant with PT schema checker.
Weights and variances are updated inplace by the kernel
*************************************************************************************/
Tensor& optimizer_adagrad_hpu(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_KERNEL_BEGIN;

  size_t device_id = gradients[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients[0].scalar_type();
  std::string node_type = "optimizer_adagrad_bwd_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  OptimizerFusedAdagradOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights),
      IValue(variances),
      IValue(epoch_num),
      IValue(lr),
      IValue(wd),
      IValue(lrd),
      IValue(epsilon)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs;
  auto num_params = static_cast<int>(gradients.size());

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(gradients[j]);
  }

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(weights[j]);
  }

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(variances[j]);
  }

  pt_inputs.push_back(epoch_num);
  pt_inputs.push_back(lr);

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);

    std::vector<at::Tensor> pt_outputs;
    for (auto j = 0; j < num_params; j++) {
      pt_outputs.push_back(weights[j]);
      pt_outputs.push_back(variances[j]);
    }

    Op.SetPTOutputs(pt_outputs);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return lr;
}

// SGD Optimizer
void OptimizerSGDOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs for optimizer SGD operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isDouble(), "Input arg4 type expected to be float");
  TORCH_CHECK(inputs[4].isDouble(), "Input arg5 type expected to be float");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isBool(), "Input arg7 type expected to be bool");

  auto gradients = inputs[0].toTensor();
  auto weights = inputs[1].toTensor();
  auto lr = inputs[2].toTensor();

  ns_OptimizerSGD::Params params;
  params.wd = inputs[3].toDouble();
  params.mom = inputs[4].toDouble();
  params.damp = inputs[5].toDouble();
  params.nesterov = inputs[6].toBool();

  // execute in-place for weights
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1]));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  PT_KERNEL_END;
}

void OptimizerFusedSGDOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  PT_KERNEL_BEGIN;

  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs for optimizer fused SGD operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isDouble(), "Input arg4 type expected to be float");
  TORCH_CHECK(inputs[4].isDouble(), "Input arg5 type expected to be float");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isBool(), "Input arg7 type expected to be bool");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto lr = inputs[2].toTensor();

  auto num_params = static_cast<int>(gradients.size());

  torch::jit::Stack stack;
  size_t device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  for (auto i = 0; i < num_params; i++) {
    OptimizerSGDOperator op(device_id, scalar_type);
    auto& syn_grad = op.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    auto& syn_wt =
        op.SetSynapseInput(std::move(p_context_->syn_inputs_[num_params + i]));
    auto& syn_lr =
        op.SetSynapseInput(std::move(p_context_->syn_inputs_[2 * num_params]));

    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(inputs[2]);
    stack.emplace_back(inputs[3]);
    stack.emplace_back(inputs[4]);
    stack.emplace_back(inputs[5]);
    stack.emplace_back(inputs[6]);

    op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    stack.clear();

    p_context_->syn_inputs_[i] = std::move(syn_grad);
    p_context_->syn_inputs_[num_params + i] = std::move(syn_wt);
    p_context_->syn_inputs_[2 * num_params] = std::move(syn_lr);

    p_context_->syn_outputs_.emplace_back(std::move(op.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(op.GetOutputs()[0]);

  } // for (auto i = 0;i < num_params;i++)
  PT_KERNEL_END;
}

/*************************************************************************************
@brief - Implements custom fused adgrad optimizer for SGD parameters
@param[in] - gradients - TensorList of gradients tensors FP32, 2D
@param[in, out] - weights - TensorList of gradients tensors FP32, 2D
@param[in] - lr - learning rate - FP32, 1D
@param[in] - wd -  weight deca - FP32
@param[in] - mom - momentum factor - FP32
@param[in] - damp - dampening for momentum - FP32
@param[in] - nesterov - enables Nesterov momentum - BOOL

@param[out] - lr - This is dummy output to be compliant with PT schema checker.
Weights are updated inplace by the kernel
*************************************************************************************/
Tensor& optimizer_sgd_hpu(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_KERNEL_BEGIN;

  size_t device_id = gradients[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients[0].scalar_type();
  std::string node_type =
      "optimizer_sgd_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  OptimizerFusedSGDOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights),
      IValue(lr),
      IValue(wd),
      IValue(mom),
      IValue(damp),
      IValue(nesterov)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs;
  auto num_params = static_cast<int>(gradients.size());

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(gradients[j]);
  }

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(weights[j]);
  }

  pt_inputs.push_back(lr);

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);

    std::vector<at::Tensor> pt_outputs;
    for (auto j = 0; j < num_params; j++) {
      pt_outputs.push_back(weights[j]);
    }

    Op.SetPTOutputs(pt_outputs);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);

    Op.AllocateAndAddSynapseNode(graph, stack, {true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return lr;
}

void OptimizerSGDMomentumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs for optimizer SGD operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input arg1 type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input arg2 type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input arg3 type expected to be tensor");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");
  TORCH_CHECK(inputs[8].isBool(), "Input arg9 type expected to be bool");

  auto gradients = inputs[0].toTensor();
  auto weights = inputs[1].toTensor();
  auto momentum = inputs[2].toTensor();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();

  ns_OptimizerSGD::Params params;
  params.wd = inputs[5].toDouble();
  params.mom = inputs[6].toDouble();
  params.damp = inputs[7].toDouble();
  params.nesterov = inputs[8].toBool();

  // execute in-place for weights & momentum
  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[1]));

  auto weights_in = inputs[1].toTensor();
  p_context_->pt_outputs_.emplace_back(weights_in);

  p_context_->syn_outputs_.emplace_back(
      habana_helpers::duplicate_tensor_in_memory_section(
          p_context_->syn_inputs_[2]));

  auto momentum_in = inputs[2].toTensor();
  p_context_->pt_outputs_.emplace_back(momentum_in);

  AddNodeToSynapseGraph(graph, &params, sizeof(params));
  PT_KERNEL_END;
}

void OptimizerFusedSGDMomentumOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    std::vector<bool> is_output_persistent) {
  PT_KERNEL_BEGIN;
  TORCH_CHECK(
      inputs.size() == 9,
      "Incorrect size of inputs for optimizer fused SGD operator");
  TORCH_CHECK(
      inputs[0].isTensorList(), "Input arg1 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[1].isTensorList(), "Input arg2 type expected to be tensorlist");
  TORCH_CHECK(
      inputs[2].isTensorList(), "Input arg3 type expected to be tensorlist");
  TORCH_CHECK(inputs[3].isTensor(), "Input arg4 type expected to be tensor");
  TORCH_CHECK(inputs[4].isTensor(), "Input arg5 type expected to be tensor");
  TORCH_CHECK(inputs[5].isDouble(), "Input arg6 type expected to be float");
  TORCH_CHECK(inputs[6].isDouble(), "Input arg7 type expected to be float");
  TORCH_CHECK(inputs[7].isDouble(), "Input arg8 type expected to be float");
  TORCH_CHECK(inputs[8].isBool(), "Input arg9 type expected to be bool");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto momentum = inputs[2].toTensorList();
  auto epoch_num = inputs[3].toTensor();
  auto lr = inputs[4].toTensor();

  auto num_params = static_cast<int>(gradients.size());

  torch::jit::Stack stack;
  size_t device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();

  for (auto i = 0; i < num_params; i++) {
    OptimizerSGDMomentumOperator op(device_id, scalar_type);
    auto& syn_grad = op.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    auto& syn_wt =
        op.SetSynapseInput(std::move(p_context_->syn_inputs_[num_params + i]));
    auto& syn_momentum = op.SetSynapseInput(
        std::move(p_context_->syn_inputs_[2 * num_params + i]));
    auto& syn_epoch_num =
        op.SetSynapseInput(std::move(p_context_->syn_inputs_[3 * num_params]));
    auto& syn_lr = op.SetSynapseInput(
        std::move(p_context_->syn_inputs_[3 * num_params + 1]));

    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(momentum.get(i)));
    stack.emplace_back(inputs[3]);
    stack.emplace_back(inputs[4]);
    stack.emplace_back(inputs[5]);
    stack.emplace_back(inputs[6]);
    stack.emplace_back(inputs[7]);
    stack.emplace_back(inputs[8]);

    op.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);

    stack.clear();

    p_context_->syn_inputs_[i] = std::move(syn_grad);
    p_context_->syn_inputs_[num_params + i] = std::move(syn_wt);
    p_context_->syn_inputs_[2 * num_params + i] = std::move(syn_momentum);
    p_context_->syn_inputs_[3 * num_params] = std::move(syn_epoch_num);
    p_context_->syn_inputs_[3 * num_params + 1] = std::move(syn_lr);

    p_context_->syn_outputs_.emplace_back(std::move(op.GetSynOutputs()[0]));

    p_context_->pt_outputs_.emplace_back(op.GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(op.GetSynOutputs()[1]));
    p_context_->pt_outputs_.emplace_back(op.GetOutputs()[1]);
  } // for (auto i = 0;i < num_params;i++)

  PT_KERNEL_END;
}

/*************************************************************************************
@brief - Implements custom fused adgrad optimizer for SGD parameters
@param[in] - gradients - TensorList of gradients tensors FP32, 2D
@param[in, out] - weights - TensorList of gradients tensors FP32, 2D
@param[in, out] - momentum - TensorList of momentum variance FP32, 2D
@param[in] - epoch_num - current epoch training number - I32, 1D
@param[in] - lr - learning rate - FP32, 1D
@param[in] - wd -  weight deca - FP32
@param[in] - mom - momentum factor - FP32
@param[in] - damp - dampening for momentum - FP32
@param[in] - nesterov - enables Nesterov momentum - BOOL

@param[out] - lr - This is dummy output to be compliant with PT schema checker.
Weights and momentum are updated inplace by the kernel
*************************************************************************************/
Tensor& optimizer_sgd_momentum_hpu(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_KERNEL_BEGIN;

  size_t device_id = gradients[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients[0].scalar_type();
  std::string node_type =
      "optimizer_sgd_bwd_" + habana_helpers::name_suffix_from_type(scalar_type);

  OptimizerFusedSGDMomentumOperator Op(device_id, scalar_type);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights),
      IValue(momentum),
      IValue(epoch_num),
      IValue(lr),
      IValue(wd),
      IValue(mom),
      IValue(damp),
      IValue(nesterov)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs;
  auto num_params = static_cast<int>(gradients.size());

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(gradients[j]);
  }

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(weights[j]);
  }

  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(momentum[j]);
  }

  pt_inputs.push_back(epoch_num);
  pt_inputs.push_back(lr);

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    Op.SetPTInputs(pt_inputs);

    std::vector<at::Tensor> pt_outputs;
    for (auto j = 0; j < num_params; j++) {
      pt_outputs.push_back(weights[j]);
      pt_outputs.push_back(momentum[j]);
    }

    Op.SetPTOutputs(pt_outputs);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);

    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, {true});
    // compile and execute the graph
    Op.Compile(graph);
  }

  PT_KERNEL_END;
  return lr;
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
            "hpu::habanaOptimizerSparseAdagrad",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerSparseAdagradOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::habanaOptimizerAdamW",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerAdamwOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::habanaOptimizerFusedAdagrad",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerFusedAdagradOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::habanaOptimizerFusedSGD",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerFusedSGDOperator>(
                  device_id, node_type);
            })
        .add(
            "hpu::habanaOptimizerFusedSGDMomentum",
            [](const int device_id, c10::ScalarType node_type) {
              return std::make_shared<OptimizerFusedSGDMomentumOperator>(
                  device_id, node_type);
            });
