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
#include "habana_helpers/graph.h"
#include "habana_helpers/tensor_utils.h"
#include "habana_helpers/unused_macro.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/optimizer_lamb.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/unary_kernels.h"
#include "simple_generic_kernel.h"
#include "synapse_helpers/recipe.h"

using namespace torch;
using namespace habana;

void OptimizerLambPhase1Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 12,
      "Incorrect size of inputs for lamb optimizer ph1 graph creation call");

  auto gradients = inputs[0].toTensorList();
  auto weights = inputs[1].toTensorList();
  auto exp_avg = inputs[2].toTensorList();
  auto exp_avg_sq = inputs[3].toTensorList();
  auto global_norm = inputs[4].toTensor();
  auto beta1 = inputs[5].toScalar();
  auto beta2 = inputs[6].toScalar();
  auto beta3 = inputs[7].toScalar();
  auto epsilon = inputs[8].toScalar();
  auto bias_correction1 = inputs[9].toTensor();
  auto bias_correction2 = inputs[10].toTensor();
  auto weight_decay = inputs[11].toScalar();

  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();
  auto num_params = static_cast<int>(weights.size());
  torch::jit::Stack stack;
  for (auto i = 0; i < num_params; i++) {
    // Synapse Graph for single parameter update to be created here
    // All synapse input tensor references are there in a single std::vector
    // in following order,
    // gradients ; weights ; exp_avg ; exp_avg_sq; global_norm;
    // bias_correction1; bias_correction2

    // grad = p.grad.data.div_(clip_global_grad_norm)
    habana::DivOperator div_grad(device_id, scalar_type);
    auto& syn_in_10 =
        div_grad.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    auto& syn_in_20 = div_grad.SetSynapseInput(
        std::move(p_context_->syn_inputs_[4 * num_params]));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(global_norm));
    div_grad.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[i] = std::move(syn_in_10);
    p_context_->syn_inputs_[4 * num_params] = std::move(syn_in_20);
    stack.clear();

    // exp_avg.mul_(beta1).add_(grad, alpha=beta3)
    habana::MulInplaceOperator mul_exp_avg(device_id, scalar_type);
    auto& syn_in_11 = mul_exp_avg.SetSynapseInput(
        std::move(p_context_->syn_inputs_[2 * num_params + i]));
    stack.emplace_back(IValue(exp_avg.get(i)));
    stack.emplace_back(IValue(beta1));
    mul_exp_avg.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[2 * num_params + i] = std::move(syn_in_11);
    stack.clear();

    habana::AddInplaceOperator add_exp_avg(device_id, scalar_type);
    auto& syn_in_12 =
        add_exp_avg.SetSynapseInput(std::move(mul_exp_avg.GetSynOutputs()[0]));
    auto& syn_in_22 =
        add_exp_avg.SetSynapseInput(std::move(div_grad.GetSynOutputs()[0]));
    stack.emplace_back(IValue(mul_exp_avg.GetOutputs()[0]));
    stack.emplace_back(IValue(div_grad.GetOutputs()[0]));
    stack.emplace_back(IValue(beta3));
    add_exp_avg.AllocateAndAddSynapseNode(graph, stack, false);
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
    UNUSED auto& syn_in_24 =
        addcmul_exp_avg_sq.SetSynapseInput(std::move(syn_in_22));
    // Internally we are going to use "pow" instead of "mul",
    // therefore 3rd synapse tensor will be unused. We can give
    // a dummy tensor
    UNUSED auto& syn_in_34 = addcmul_exp_avg_sq.SetSynapseInput(
        std::move(habana_helpers::create_tensor(
            div_grad.GetOutputs()[0],
            graph.get_graph_handle(),
            true,
            c10::nullopt)));
    stack.emplace_back(IValue(mul_exp_avg_sq.GetOutputs()[0]));
    stack.emplace_back(IValue(div_grad.GetOutputs()[0]));
    stack.emplace_back(IValue(div_grad.GetOutputs()[0]));
    stack.emplace_back(IValue(Scalar(1.0 - beta2.toFloat())));
    addcmul_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // exp_avg = exp_avg_.div(bias_correction1)
    // exp_avg_sq = exp_avg_sq_.div(bias_correction2)
    habana::DivOperator div_exp_avg(device_id, scalar_type);
    auto& syn_in_15 =
        div_exp_avg.SetSynapseInput(std::move(add_exp_avg.GetSynOutputs()[0]));
    auto& syn_in_25 = div_exp_avg.SetSynapseInput(
        std::move(p_context_->syn_inputs_[4 * num_params + 1]));
    stack.emplace_back(IValue(add_exp_avg.GetOutputs()[0]));
    stack.emplace_back(IValue(bias_correction1));
    div_exp_avg.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[4 * num_params + 1] = std::move(syn_in_25);
    stack.clear();
    habana::DivOperator div_exp_avg_sq(device_id, scalar_type);
    auto& syn_in_16 = div_exp_avg_sq.SetSynapseInput(
        std::move(addcmul_exp_avg_sq.GetSynOutputs()[0]));
    auto& syn_in_26 = div_exp_avg_sq.SetSynapseInput(
        std::move(p_context_->syn_inputs_[4 * num_params + 2]));
    stack.emplace_back(IValue(addcmul_exp_avg_sq.GetOutputs()[0]));
    stack.emplace_back(IValue(bias_correction2));
    div_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[4 * num_params + 2] = std::move(syn_in_26);
    stack.clear();

    // denom = exp_avg_sq.sqrt().add_(group["eps"])
    // we will actually do "add" instead of "add_". Inplace not strictly
    // required here
    SqrtOperator sqrt_exp_avg_sq(device_id, scalar_type);
    UNUSED auto& syn_in_17 = sqrt_exp_avg_sq.SetSynapseInput(
        std::move(div_exp_avg_sq.GetSynOutputs()[0]));
    stack.emplace_back(IValue(div_exp_avg_sq.GetOutputs()[0]));
    sqrt_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    habana::AddOperator add_exp_avg_sq(device_id, scalar_type);
    UNUSED auto& syn_in_18 = add_exp_avg_sq.SetSynapseInput(
        std::move(sqrt_exp_avg_sq.GetSynOutputs()[0]));
    stack.emplace_back(IValue(sqrt_exp_avg_sq.GetOutputs()[0]));
    stack.emplace_back(IValue(epsilon));
    stack.emplace_back(IValue(1.0));
    add_exp_avg_sq.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    if (weight_decay.toFloat() != 0.0) {
      // adam_step = torch.div(exp_avg, denom)
      habana::DivOperator div_wt(device_id, scalar_type);
      UNUSED auto& syn_in_19 =
          div_wt.SetSynapseInput(std::move(div_exp_avg.GetSynOutputs()[0]));
      UNUSED auto& syn_in_29 =
          div_wt.SetSynapseInput(std::move(add_exp_avg_sq.GetSynOutputs()[0]));
      stack.emplace_back(IValue(div_exp_avg.GetOutputs()[0]));
      stack.emplace_back(IValue(add_exp_avg_sq.GetOutputs()[0]));
      div_wt.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      // adam_step.add_(p.data, alpha=group['weight_decay'])
      habana::AddOperator add_wt(device_id, scalar_type);
      UNUSED auto& syn_in_100 =
          add_wt.SetSynapseInput(std::move(div_wt.GetSynOutputs()[0]));
      auto& syn_in_200 = add_wt.SetSynapseInput(
          std::move(p_context_->syn_inputs_[1 * num_params + i]));
      stack.emplace_back(IValue(div_wt.GetOutputs()[0]));
      stack.emplace_back(IValue(weights.get(i)));
      stack.emplace_back(IValue(weight_decay));
      add_wt.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
      p_context_->syn_inputs_[1 * num_params + i] = std::move(syn_in_200);
      stack.clear();

      // adam_norm = adam_step.norm()
      NormOperator norm_adam_step(device_id, scalar_type);
      auto& syn_in_101 =
          norm_adam_step.SetSynapseInput(std::move(add_wt.GetSynOutputs()[0]));
      stack.emplace_back(IValue(add_wt.GetOutputs()[0]));
      stack.emplace_back(IValue(2.0));
      norm_adam_step.AllocateAndAddSynapseNode(
          graph, stack, is_output_persistent);
      stack.clear();
      p_context_->syn_outputs_.emplace_back(std::move(syn_in_101));
      p_context_->pt_outputs_.emplace_back(add_wt.GetOutputs()[0]);
      p_context_->syn_outputs_.emplace_back(
          std::move(norm_adam_step.GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(norm_adam_step.GetOutputs()[0]);
    } else {
      // adam_step = torch.div(exp_avg, denom)
      habana::DivOperator div_wt(device_id, scalar_type);
      UNUSED auto& syn_in_19 =
          div_wt.SetSynapseInput(std::move(div_exp_avg.GetSynOutputs()[0]));
      UNUSED auto& syn_in_29 =
          div_wt.SetSynapseInput(std::move(add_exp_avg_sq.GetSynOutputs()[0]));
      stack.emplace_back(IValue(div_exp_avg.GetOutputs()[0]));
      stack.emplace_back(IValue(add_exp_avg_sq.GetOutputs()[0]));
      div_wt.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
      stack.clear();

      NormOperator norm_adam_step(device_id, scalar_type);
      auto& syn_in_101 =
          norm_adam_step.SetSynapseInput(std::move(div_wt.GetSynOutputs()[0]));
      stack.emplace_back(IValue(div_wt.GetOutputs()[0]));
      stack.emplace_back(IValue(2.0));
      norm_adam_step.AllocateAndAddSynapseNode(
          graph, stack, is_output_persistent);
      stack.clear();
      p_context_->syn_outputs_.emplace_back(std::move(syn_in_101));
      p_context_->pt_outputs_.emplace_back(div_wt.GetOutputs()[0]);
      p_context_->syn_outputs_.emplace_back(
          std::move(norm_adam_step.GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(norm_adam_step.GetOutputs()[0]);
    }

    // weight_norm = p.data.norm()
    NormOperator norm_wt(device_id, scalar_type);
    auto& syn_in_102 = norm_wt.SetSynapseInput(
        std::move(p_context_->syn_inputs_[1 * num_params + i]));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(2.0));
    norm_wt.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_inputs_[1 * num_params + i] = std::move(syn_in_102);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(norm_wt.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(norm_wt.GetOutputs()[0]);

    // Note that these outputs are being filled just to keep GC
    // runtime happy No need to return these since updates on
    // weights, exp_avg, exp_avg_sq are all inplace
    p_context_->syn_outputs_.emplace_back(std::move(syn_in_12));
    p_context_->pt_outputs_.emplace_back(mul_exp_avg.GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(syn_in_15));
    p_context_->pt_outputs_.emplace_back(add_exp_avg.GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(syn_in_14));
    p_context_->pt_outputs_.emplace_back(mul_exp_avg_sq.GetOutputs()[0]);

    p_context_->syn_outputs_.emplace_back(std::move(syn_in_16));
    p_context_->pt_outputs_.emplace_back(addcmul_exp_avg_sq.GetOutputs()[0]);
  }
}

std::tuple<std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>>
optimizer_lamb_phase1_hpu(
    const std::vector<at::Tensor>& gradient_vec,
    std::vector<at::Tensor>& weight_vec,
    std::vector<at::Tensor>& exp_avg_vec,
    std::vector<at::Tensor>& exp_avg_sq_vec,
    const Tensor& clip_global_grad_norm,
    const int grad_averaging,
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

  // TBD: Encapsulate all pre-processing on inputs (before graph creation)
  // into a separate function. Will be needed if this custom optimizer is
  // used in Lazy mode.
  float bias_correction1 = 1.0, bias_correction2 = 1.0;
  if (bias_correction) {
    bias_correction1 = 1.0 - std::pow(beta1, step);
    bias_correction2 = 1.0 - std::pow(beta2, step);
  }

  float beta3 = 1.0;
  if (grad_averaging) {
    beta3 = 1 - beta1;
  }

  size_t device_id = gradients[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = gradients[0].scalar_type();
  std::string node_type = "optimizer_lamb_ph1_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  OptimizerLambPhase1Operator Op(device_id, scalar_type);

  // only bias_corrections need to be converted to tensors since these may
  // change every iteration. not needed for beta(s), epsilon that never change
  // or weight_decay that does not change frequently
  auto bias_correction1_t = habana_helpers::GenerateAndCopyTensorToHPU(
      weights[0], bias_correction1, true);
  auto bias_correction2_t = habana_helpers::GenerateAndCopyTensorToHPU(
      weights[0], bias_correction2, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(gradients),
      IValue(weights),
      IValue(exp_avg),
      IValue(exp_avg_sq),
      IValue(clip_global_grad_norm),
      IValue(beta1),
      IValue(beta2),
      IValue(beta3),
      IValue(epsilon),
      IValue(bias_correction1_t),
      IValue(bias_correction2_t),
      IValue(weight_decay)};

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
    pt_inputs.push_back(exp_avg[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(exp_avg_sq[j]);
  }
  pt_inputs.push_back(clip_global_grad_norm);
  pt_inputs.push_back(bias_correction1_t);
  pt_inputs.push_back(bias_correction2_t);

  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    std::vector<at::Tensor> pt_outputs;
    for (auto j = 0; j < num_params; j++) {
      pt_outputs.push_back(at::empty_like(weights[j])); // adam_step
      pt_outputs.push_back(at::empty_like(weights[j])); // adam_norm
      pt_outputs.push_back(at::empty_like(weights[j])); // weight_norm
      pt_outputs.push_back(exp_avg[j]);
      pt_outputs.push_back(exp_avg[j]);
      pt_outputs.push_back(exp_avg_sq[j]);
      pt_outputs.push_back(exp_avg_sq[j]);
    }
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

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(
      out.size() == static_cast<unsigned int>(7 * num_params),
      "Incorrect size of outputs");
  // TBD: Currently Norm output is of same shape as Norm input
  // due to TPC kernel limitation, therefore to get correct Norm
  // output shape we manipulate tensor meta-data. This can be
  // removed once we have updated TPC kernel for Norm.
  std::vector<Tensor> weight_norm, adam_norm, adam_step;
  adam_step.push_back(out[0]);
  out[1].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  adam_norm.push_back(out[1]);
  out[2].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
  weight_norm.push_back(out[2]);
  for (auto j = 1; j < num_params; j++) {
    adam_step.push_back(out[7 * j]);
    out[7 * j + 1].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    adam_norm.push_back(out[7 * j + 1]);
    out[7 * j + 2].unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1});
    weight_norm.push_back(out[7 * j + 2]);
  }

  PT_KERNEL_END;
  return std::tie(weight_norm, adam_norm, adam_step);
}

void OptimizerLambPhase2Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 8,
      "Incorrect size of inputs for lamb optimizer ph2 graph creation call");

  auto weights = inputs[0].toTensorList();
  auto adam_norm = inputs[1].toTensorList();
  auto weight_norm = inputs[2].toTensorList();
  auto adam_step = inputs[3].toTensorList();
  auto trust_ratio = inputs[4].toTensorList();
  auto nstep = inputs[5].toTensor();
  auto weight_decay = inputs[6].toScalar();
  auto use_lamb = inputs[7].toScalar();

  auto device_id = weights.get(0).device().index();
  auto scalar_type = weights.get(0).scalar_type();
  auto num_params = static_cast<int>(weights.size());
  torch::jit::Stack stack;
  for (auto i = 0; i < num_params; i++) {
    // Synapse Graph for single parameter update to be created here
    // All synapse input tensor references are there in a single std::vector
    // arranged as follows,
    // weights ; adam_norm ; weight_norm; adam_step; trust_ratio; nstep

    MulOperator mul4(device_id, scalar_type);
    if ((weight_decay.toFloat() != 0.0) || use_lamb.toInt()) {
      // weight_norm / adam_norm
      DivOperator div_lp(device_id, scalar_type);
      auto& syn_in_10 = div_lp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[2 * num_params + i]));
      auto& syn_in_20 = div_lp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[num_params + i]));
      stack.emplace_back(IValue(weight_norm.get(i)));
      stack.emplace_back(IValue(adam_norm.get(i)));
      div_lp.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[2 * num_params + i] = std::move(syn_in_10);
      p_context_->syn_inputs_[num_params + i] = std::move(syn_in_20);
      stack.clear();

      // weight_norm + adam_norm
      AddOperator add1_lp(device_id, scalar_type);
      auto& syn_in_11 = add1_lp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[2 * num_params + i]));
      auto& syn_in_21 = add1_lp.SetSynapseInput(
          std::move(p_context_->syn_inputs_[num_params + i]));
      stack.emplace_back(IValue(weight_norm.get(i)));
      stack.emplace_back(IValue(adam_norm.get(i)));
      stack.emplace_back(IValue(1.0));
      add1_lp.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[2 * num_params + i] = std::move(syn_in_11);
      p_context_->syn_inputs_[num_params + i] = std::move(syn_in_21);
      stack.clear();

      // mask = (weight_norm + adam_norm == 0)
      EqOperator eq1_lp(device_id, scalar_type);
      eq1_lp.SetSynapseInput(std::move(add1_lp.GetSynOutputs()[0]));
      stack.emplace_back(IValue(add1_lp.GetOutputs()[0]));
      stack.emplace_back(IValue(0.0));
      eq1_lp.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      std::string node_type = "cast_i8_to_f32";
      CastOperator cast1(device_id, node_type);
      cast1.SetSynapseInput(std::move(eq1_lp.GetSynOutputs()[0]));
      stack.emplace_back(IValue(eq1_lp.GetOutputs()[0]));
      stack.emplace_back(IValue(c10::ScalarType::Float));
      cast1.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      // mul1 = mask * trust_ratio(=1)
      MulOperator mul1(device_id, scalar_type);
      auto& syn_in_30 =
          mul1.SetSynapseInput(std::move(cast1.GetSynOutputs()[0]));
      auto& syn_in_31 = mul1.SetSynapseInput(
          std::move(p_context_->syn_inputs_[4 * num_params + i]));
      stack.emplace_back(IValue(cast1.GetOutputs()[0]));
      stack.emplace_back(IValue(trust_ratio.get(i)));
      mul1.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[4 * num_params + i] = std::move(syn_in_31);
      stack.clear();

      // imask = (mask == 0)
      EqOperator eq2_lp(device_id, scalar_type);
      eq2_lp.SetSynapseInput(std::move(syn_in_30));
      stack.emplace_back(IValue(cast1.GetOutputs()[0]));
      stack.emplace_back(IValue(0));
      eq2_lp.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      node_type = "cast_i8_to_f32";
      CastOperator cast2(device_id, node_type);
      cast2.SetSynapseInput(std::move(eq2_lp.GetSynOutputs()[0]));
      stack.emplace_back(IValue(eq2_lp.GetOutputs()[0]));
      stack.emplace_back(IValue(c10::ScalarType::Float));
      cast2.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      // mul2 = imask * weight_norm / adam_norm
      MulOperator mul2(device_id, scalar_type);
      mul2.SetSynapseInput(std::move(cast2.GetSynOutputs()[0]));
      mul2.SetSynapseInput(std::move(div_lp.GetSynOutputs()[0]));
      stack.emplace_back(IValue(cast2.GetOutputs()[0]));
      stack.emplace_back(IValue(div_lp.GetOutputs()[0]));
      mul2.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      // trust_ratio = mask * trust_ratio(=1) + imask * weight_norm / adam_norm
      AddOperator add2_lp(device_id, scalar_type);
      add2_lp.SetSynapseInput(std::move(mul1.GetSynOutputs()[0]));
      add2_lp.SetSynapseInput(std::move(mul2.GetSynOutputs()[0]));
      stack.emplace_back(IValue(mul1.GetOutputs()[0]));
      stack.emplace_back(IValue(mul2.GetOutputs()[0]));
      stack.emplace_back(IValue(1.0));
      add2_lp.AllocateAndAddSynapseNode(graph, stack, false);
      stack.clear();

      // trust_ratio * -step
      MulOperator mul3(device_id, scalar_type);
      mul3.SetSynapseInput(std::move(add2_lp.GetSynOutputs()[0]));
      auto& syn_in_400 = mul3.SetSynapseInput(
          std::move(p_context_->syn_inputs_[5 * num_params]));
      stack.emplace_back(IValue(add2_lp.GetOutputs()[0]));
      stack.emplace_back(IValue(nstep));
      mul3.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[5 * num_params] = std::move(syn_in_400);
      stack.clear();

      // trust_ratio * -step * adam_step
      mul4.SetSynapseInput(std::move(mul3.GetSynOutputs()[0]));
      auto& syn_in_40 = mul4.SetSynapseInput(
          std::move(p_context_->syn_inputs_[3 * num_params + i]));
      stack.emplace_back(IValue(mul3.GetOutputs()[0]));
      stack.emplace_back(IValue(adam_step.get(i)));
      mul4.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[3 * num_params + i] = std::move(syn_in_40);
      stack.clear();

    } else {
      // -step * adam_step
      auto& syn_in_40 = mul4.SetSynapseInput(
          std::move(p_context_->syn_inputs_[3 * num_params + i]));
      auto& syn_in_400 = mul4.SetSynapseInput(
          std::move(p_context_->syn_inputs_[5 * num_params]));
      stack.emplace_back(IValue(adam_step.get(i)));
      stack.emplace_back(IValue(nstep));
      mul4.AllocateAndAddSynapseNode(graph, stack, false);
      p_context_->syn_inputs_[3 * num_params + i] = std::move(syn_in_40);
      p_context_->syn_inputs_[5 * num_params] = std::move(syn_in_400);
      stack.clear();
    }

    // p.data.add_(adam_step)
    AddInplaceOperator add3_lp(device_id, scalar_type);
    auto& syn_in_50 =
        add3_lp.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    add3_lp.SetSynapseInput(std::move(mul4.GetSynOutputs()[0]));
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(mul4.GetOutputs()[0]));
    stack.emplace_back(IValue(1.0));
    add3_lp.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
    p_context_->syn_inputs_[i] = std::move(syn_in_50);
    stack.clear();

    // Note that these outputs are being filled just to keep GC
    // runtime happy No need to return these since updates on
    // weights are all inplace
    p_context_->syn_outputs_.emplace_back(
        std::move(add3_lp.GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(add3_lp.GetOutputs()[0]);
  }
}

void optimizer_lamb_phase2_hpu(
    std::vector<at::Tensor>& weight_vec,
    const std::vector<at::Tensor>& adam_norm_vec,
    const std::vector<at::Tensor>& weight_norm_vec,
    const std::vector<at::Tensor>& adam_step_vec,
    const std::vector<at::Tensor>& trust_ratio_vec,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_KERNEL_BEGIN;

  TensorList weights(weight_vec);
  TensorList adam_norm(adam_norm_vec);
  TensorList weight_norm(weight_norm_vec);
  TensorList adam_step(adam_step_vec);
  TensorList trust_ratio(trust_ratio_vec);

  size_t device_id = weights[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = weights[0].scalar_type();
  std::string node_type = "optimizer_lamb_ph2_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  OptimizerLambPhase2Operator Op(device_id, scalar_type);

  // TBD: Encapsulate all pre-processing on inputs (before graph creation)
  // into a separate function. Will be needed if this custom optimizer is
  // used in Lazy mode.

  // only -step needs to be converted to tensors since this may
  // change every iteration. not needed for use_lamb, that never changes
  // or weight_decay that does not change frequently
  auto nstep_t =
      habana_helpers::GenerateAndCopyTensorToHPU(weights[0], -step, true);

  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(weights),
      IValue(adam_norm),
      IValue(weight_norm),
      IValue(adam_step),
      IValue(trust_ratio),
      IValue(nstep_t),
      IValue(weight_decay),
      IValue(use_lamb)};

  // Assign Inputs to the Operator
  std::vector<at::Tensor> pt_inputs;
  std::vector<at::Tensor> pt_outputs;
  auto num_params = static_cast<int>(weights.size());
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(weights[j]);
    pt_outputs.push_back(weights[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(adam_norm[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(weight_norm[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(adam_step[j]);
  }
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(trust_ratio[j]);
  }
  pt_inputs.push_back(nstep_t);

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
}

void OptNormFusedNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    bool is_output_persistent) {
  TORCH_CHECK(
      inputs.size() == 3,
      "Incorrect size of inputs expected for OptLambFusedNorm Operator");

  auto gradients = inputs[0].toTensorList();
  auto max_grad_norm = inputs[1].toScalar();
  auto clip_norm = inputs[2].toTensor();

  auto device_id = gradients.get(0).device().index();
  auto scalar_type = gradients.get(0).scalar_type();
  auto num_params = static_cast<int>(gradients.size());

  torch::jit::Stack stack;
  std::vector<Tensor> cat_input;
  CatOperator cat_grads(device_id, scalar_type);
  std::vector<int64_t> shape{1, 1};
  for (auto i = 0; i < num_params; i++) {
    // Add node to compute norm on each gradient tensor
    PowOperator pow_lp(device_id, scalar_type);
    auto& syn_in_10 =
        pow_lp.SetSynapseInput(std::move(p_context_->syn_inputs_[i]));
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(2.0));
    pow_lp.AllocateAndAddSynapseNode(graph, stack, false);
    p_context_->syn_inputs_[i] = std::move(syn_in_10);
    stack.clear();

    // add node to compute reduce_sum
    SumOperator sum_lp(device_id, scalar_type);
    sum_lp.SetSynapseInput(std::move(pow_lp.GetSynOutputs()[0]));
    stack.emplace_back(IValue(pow_lp.GetOutputs()[0]));
    stack.emplace_back(IValue(scalar_type));
    sum_lp.AllocateAndAddSynapseNode(graph, stack, false);
    stack.clear();

    // Unsqueeze sum output (tensor of shape {1}) to new tensor of shape
    // {1,1} in prep for cat
    ReshapeOperator reshape_lp(device_id, scalar_type);
    reshape_lp.SetSynapseInput(std::move(sum_lp.GetSynOutputs()[0]));
    stack.emplace_back(IValue(sum_lp.GetOutputs()[0]));
    stack.emplace_back(IValue(shape));
    reshape_lp.AllocateAndAddSynapseNode(graph, stack, false);
    // each unsqueezed grad_norm connected to cat node
    cat_input.push_back(reshape_lp.GetOutputs()[0]);
    cat_grads.SetSynapseInput(std::move(reshape_lp.GetSynOutputs()[0]));
    stack.clear();
  }

  // grads are concatened into a single big tensor of shape
  // {num_params,1}
  stack.emplace_back(IValue(cat_input));
  stack.emplace_back(IValue(0));
  cat_grads.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // add node to compute reduce_sum
  SumOperator sum_final(device_id, scalar_type);
  sum_final.SetSynapseInput(std::move(cat_grads.GetSynOutputs()[0]));
  stack.emplace_back(IValue(cat_grads.GetOutputs()[0]));
  stack.emplace_back(IValue(scalar_type));
  sum_final.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // global_grad_norm = global_grad_norm.sqrt()
  SqrtOperator sqrt_final(device_id, scalar_type);
  sqrt_final.SetSynapseInput(std::move(sum_final.GetSynOutputs()[0]));
  stack.emplace_back(IValue(sum_final.GetOutputs()[0]));
  sqrt_final.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // if global_grad_norm > max_grad_norm:
  //    clip_global_grad_norm = global_grad_norm / max_grad_norm
  // else:
  //    clip_global_grad_norm = 1.0

  // global_grad_norm / max_grad_norm
  DivOperator div_final(device_id, scalar_type);
  auto& syn_in_20 =
      div_final.SetSynapseInput(std::move(sqrt_final.GetSynOutputs()[0]));
  stack.emplace_back(IValue(sqrt_final.GetOutputs()[0]));
  stack.emplace_back(IValue(max_grad_norm));
  div_final.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // mask = global_grad_norm < max_grad_norm
  LtOperator lt_final(device_id, scalar_type);
  lt_final.SetSynapseInput(std::move(syn_in_20));
  stack.emplace_back(IValue(sqrt_final.GetOutputs()[0]));
  stack.emplace_back(IValue(max_grad_norm));
  lt_final.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  std::string node_type = "cast_i8_to_f32";
  CastOperator cast1(device_id, node_type);
  cast1.SetSynapseInput(std::move(lt_final.GetSynOutputs()[0]));
  stack.emplace_back(IValue(lt_final.GetOutputs()[0]));
  stack.emplace_back(IValue(c10::ScalarType::Float));
  cast1.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // mul1 = mask * clip_norm(=1)
  MulOperator mul1(device_id, scalar_type);
  auto& syn_in_30 = mul1.SetSynapseInput(std::move(cast1.GetSynOutputs()[0]));
  auto& syn_in_31 =
      mul1.SetSynapseInput(std::move(p_context_->syn_inputs_[num_params]));
  stack.emplace_back(IValue(cast1.GetOutputs()[0]));
  stack.emplace_back(IValue(clip_norm));
  mul1.AllocateAndAddSynapseNode(graph, stack, false);
  p_context_->syn_inputs_[num_params] = std::move(syn_in_31);
  stack.clear();

  // imask = (mask == 0)
  EqOperator eq_final(device_id, scalar_type);
  eq_final.SetSynapseInput(std::move(syn_in_30));
  stack.emplace_back(IValue(cast1.GetOutputs()[0]));
  stack.emplace_back(IValue(0));
  eq_final.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  node_type = "cast_i8_to_f32";
  CastOperator cast2(device_id, node_type);
  cast2.SetSynapseInput(std::move(eq_final.GetSynOutputs()[0]));
  stack.emplace_back(IValue(eq_final.GetOutputs()[0]));
  stack.emplace_back(IValue(c10::ScalarType::Float));
  cast2.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // mul2 = imask * (global_grad_norm / max_grad_norm)
  MulOperator mul2(device_id, scalar_type);
  mul2.SetSynapseInput(std::move(cast2.GetSynOutputs()[0]));
  mul2.SetSynapseInput(std::move(div_final.GetSynOutputs()[0]));
  stack.emplace_back(IValue(cast2.GetOutputs()[0]));
  stack.emplace_back(IValue(div_final.GetOutputs()[0]));
  mul2.AllocateAndAddSynapseNode(graph, stack, false);
  stack.clear();

  // out = mask * clip_norm(=1) + imask * (global_grad_norm / max_grad_norm)
  AddOperator add(device_id, scalar_type);
  add.SetSynapseInput(std::move(mul1.GetSynOutputs()[0]));
  add.SetSynapseInput(std::move(mul2.GetSynOutputs()[0]));
  stack.emplace_back(IValue(mul1.GetOutputs()[0]));
  stack.emplace_back(IValue(mul2.GetOutputs()[0]));
  stack.emplace_back(IValue(1.0));
  add.AllocateAndAddSynapseNode(graph, stack, is_output_persistent);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(std::move(add.GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(add.GetOutputs()[0]);
}

Tensor optimizer_lamb_fused_norm_hpu(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_KERNEL_BEGIN;

  auto clip_norm = torch::ones(1).to(torch::kHABANA);
  size_t device_id = grad[0].device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);
  auto scalar_type = grad[0].scalar_type();
  std::string node_type = "opt_lamb_fused_norm_" +
      habana_helpers::name_suffix_from_type(scalar_type);
  OptNormFusedNormOperator Op(device_id, scalar_type);
  // Build Params for the graph
  std::vector<c10::IValue> stack = {
      IValue(grad), IValue(max_grad_norm), IValue(clip_norm)};

  std::vector<at::Tensor> pt_inputs;
  auto num_params = static_cast<int>(grad.size());
  pt_inputs.reserve(num_params);
  for (auto j = 0; j < num_params; j++) {
    pt_inputs.push_back(grad[j]);
  }
  pt_inputs.push_back(clip_norm);
  size_t key = Op.GetRecipeKey(node_type, stack, true);
  if (device.get_recipe_handle_cache().isCached(key)) {
    PT_KERNEL_DEBUG("Cache hit key:", key);
    auto output = habana_helpers::createPTTensor(
        grad[0], {1}, grad[0].options(), grad[0].suggest_memory_format(), true);
    Op.SetPTInputs(pt_inputs);
    Op.SetPTOutput(output);
    Op.Execute(key);
  } else {
    // Create Graph
    auto graph = habana_helpers::create_graph(device_id, node_type);
    Op.AllocateSynapseInputs(graph, pt_inputs, true);
    Op.AllocateAndAddSynapseNode(graph, stack, true);
    // compile and execute the graph
    Op.Compile(graph);
  }

  std::vector<at::Tensor> out = Op.GetOutputs();
  TORCH_CHECK(out.size() == 1, "Incorrect size of outputs");

  PT_KERNEL_END;
  return out[0];
}