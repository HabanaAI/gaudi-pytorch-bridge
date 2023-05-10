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
#include <ATen/core/Reduction.h>
#include <perf_lib_layer_params.h>

#include "backend/create_pt_tensor.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_helpers/logging.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/optimizer_lamb.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/unary_kernels.h"

using namespace torch;
using namespace habana;

void OptimizerLambPhase1Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
    auto div_grad = make_operator<habana::DivOperator>(device_id, scalar_type);
    div_grad->SetSynapseInput(p_context_->syn_inputs_[i]);
    div_grad->SetSynapseInput(p_context_->syn_inputs_[4 * num_params]);
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(global_norm));
    div_grad->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // exp_avg.mul_(beta1).add_(grad, alpha=beta3)
    auto mul_exp_avg =
        make_operator<habana::MulInplaceOperator>(device_id, scalar_type);
    mul_exp_avg->SetSynapseInput(p_context_->syn_inputs_[2 * num_params + i]);
    stack.emplace_back(IValue(exp_avg.get(i)));
    stack.emplace_back(IValue(beta1));
    mul_exp_avg->AllocateAndAddSynapseNode(
        graph, stack, {output_metadata.at(3)});
    stack.clear();

    auto add_exp_avg =
        make_operator<habana::AddInplaceOperator>(device_id, scalar_type);
    add_exp_avg->SetSynapseInput(mul_exp_avg->GetSynOutputs()[0]);
    add_exp_avg->SetSynapseInput(div_grad->GetSynOutputs()[0]);
    stack.emplace_back(IValue(mul_exp_avg->GetOutputs()[0]));
    stack.emplace_back(IValue(div_grad->GetOutputs()[0]));
    stack.emplace_back(IValue(beta3));
    add_exp_avg->AllocateAndAddSynapseNode(
        graph, stack, {output_metadata.at(4)});
    stack.clear();

    // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    auto mul_exp_avg_sq =
        make_operator<habana::MulInplaceOperator>(device_id, scalar_type);
    mul_exp_avg_sq->SetSynapseInput(
        p_context_->syn_inputs_[3 * num_params + i]);
    stack.emplace_back(IValue(exp_avg_sq.get(i)));
    stack.emplace_back(IValue(beta2));
    mul_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, {output_metadata.at(5)});
    stack.clear();

    auto addcmul_exp_avg_sq =
        make_operator<habana::AddcmulInplaceOperator>(device_id, scalar_type);
    addcmul_exp_avg_sq->SetSynapseInput(mul_exp_avg_sq->GetSynOutputs()[0]);
    addcmul_exp_avg_sq->SetSynapseInput(div_grad->GetSynOutputs()[0]);
    // Internally we are going to use "pow" instead of "mul",
    // therefore 3rd synapse tensor will be unused. We can give
    // a dummy tensor
    auto syn_in_34 = habana_helpers::create_tensor(
        div_grad->GetOutputs()[0], graph, true, false, c10::nullopt);
    addcmul_exp_avg_sq->SetSynapseInput(syn_in_34);
    stack.emplace_back(IValue(mul_exp_avg_sq->GetOutputs()[0]));
    stack.emplace_back(IValue(div_grad->GetOutputs()[0]));
    stack.emplace_back(IValue(div_grad->GetOutputs()[0]));
    stack.emplace_back(IValue(Scalar(1.0 - beta2.toFloat())));
    addcmul_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, {output_metadata.at(6)});
    stack.clear();

    // exp_avg = exp_avg_.div(bias_correction1)
    // exp_avg_sq = exp_avg_sq_.div(bias_correction2)
    auto div_exp_avg =
        make_operator<habana::DivOperator>(device_id, scalar_type);
    div_exp_avg->SetSynapseInput(add_exp_avg->GetSynOutputs()[0]);
    div_exp_avg->SetSynapseInput(p_context_->syn_inputs_[4 * num_params + 1]);
    stack.emplace_back(IValue(add_exp_avg->GetOutputs()[0]));
    stack.emplace_back(IValue(bias_correction1));
    div_exp_avg->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
    auto div_exp_avg_sq =
        make_operator<habana::DivOperator>(device_id, scalar_type);
    div_exp_avg_sq->SetSynapseInput(addcmul_exp_avg_sq->GetSynOutputs()[0]);
    div_exp_avg_sq->SetSynapseInput(
        p_context_->syn_inputs_[4 * num_params + 2]);
    stack.emplace_back(IValue(addcmul_exp_avg_sq->GetOutputs()[0]));
    stack.emplace_back(IValue(bias_correction2));
    div_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // denom = exp_avg_sq.sqrt().add_(group["eps"])
    // we will actually do "add" instead of "add_". Inplace not strictly
    // required here
    auto sqrt_exp_avg_sq = make_operator<SqrtOperator>(device_id, scalar_type);
    sqrt_exp_avg_sq->SetSynapseInput(div_exp_avg_sq->GetSynOutputs()[0]);
    stack.emplace_back(IValue(div_exp_avg_sq->GetOutputs()[0]));
    sqrt_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto add_exp_avg_sq =
        make_operator<habana::AddOperator>(device_id, scalar_type);
    add_exp_avg_sq->SetSynapseInput(sqrt_exp_avg_sq->GetSynOutputs()[0]);
    stack.emplace_back(IValue(sqrt_exp_avg_sq->GetOutputs()[0]));
    stack.emplace_back(IValue(epsilon));
    stack.emplace_back(IValue(1.0));
    add_exp_avg_sq->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    if (weight_decay.toFloat() != 0.0) {
      // adam_step = torch.div(exp_avg, denom)
      auto div_wt = make_operator<habana::DivOperator>(device_id, scalar_type);
      div_wt->SetSynapseInput(div_exp_avg->GetSynOutputs()[0]);
      div_wt->SetSynapseInput(add_exp_avg_sq->GetSynOutputs()[0]);
      stack.emplace_back(IValue(div_exp_avg->GetOutputs()[0]));
      stack.emplace_back(IValue(add_exp_avg_sq->GetOutputs()[0]));
      div_wt->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      // adam_step.add_(p.data, alpha=group['weight_decay'])
      auto add_wt = make_operator<habana::AddOperator>(device_id, scalar_type);
      add_wt->SetSynapseInput(div_wt->GetSynOutputs()[0]);
      add_wt->SetSynapseInput(p_context_->syn_inputs_[1 * num_params + i]);
      stack.emplace_back(IValue(div_wt->GetOutputs()[0]));
      stack.emplace_back(IValue(weights.get(i)));
      stack.emplace_back(IValue(weight_decay));
      add_wt->AllocateAndAddSynapseNode(graph, stack, {output_metadata.at(i)});
      stack.clear();

      // adam_norm = adam_step.norm()
      auto norm_adam_step = make_operator<NormOperator>(device_id, scalar_type);
      norm_adam_step->SetSynapseInput(add_wt->GetSynOutputs()[0]);
      stack.emplace_back(IValue(add_wt->GetOutputs()[0]));
      stack.emplace_back(IValue(2.0));
      norm_adam_step->AllocateAndAddSynapseNode(
          graph, stack, {output_metadata.at(i + 1)});
      stack.clear();
      synapse_helpers::tensor& syn_out_0 = add_wt->GetSynOutputs()[0];
      p_context_->syn_outputs_.emplace_back(syn_out_0);
      p_context_->pt_outputs_.emplace_back(add_wt->GetOutputs()[0]);
      p_context_->syn_outputs_.emplace_back(
          std::move(norm_adam_step->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(norm_adam_step->GetOutputs()[0]);
    } else {
      // adam_step = torch.div(exp_avg, denom)
      auto div_wt = make_operator<habana::DivOperator>(device_id, scalar_type);
      div_wt->SetSynapseInput(div_exp_avg->GetSynOutputs()[0]);
      div_wt->SetSynapseInput(add_exp_avg_sq->GetSynOutputs()[0]);
      stack.emplace_back(IValue(div_exp_avg->GetOutputs()[0]));
      stack.emplace_back(IValue(add_exp_avg_sq->GetOutputs()[0]));
      div_wt->AllocateAndAddSynapseNode(
          graph, stack, {output_metadata.at(i + 2)});
      stack.clear();

      auto norm_adam_step = make_operator<NormOperator>(device_id, scalar_type);
      norm_adam_step->SetSynapseInput(div_wt->GetSynOutputs()[0]);
      stack.emplace_back(IValue(div_wt->GetOutputs()[0]));
      stack.emplace_back(IValue(2.0));
      norm_adam_step->AllocateAndAddSynapseNode(
          graph, stack, {output_metadata.at(i + 1)});
      stack.clear();
      synapse_helpers::tensor& div_wt_syn_out = div_wt->GetSynOutputs()[0];
      p_context_->syn_outputs_.emplace_back(div_wt_syn_out);
      p_context_->pt_outputs_.emplace_back(div_wt->GetOutputs()[0]);
      p_context_->syn_outputs_.emplace_back(
          std::move(norm_adam_step->GetSynOutputs()[0]));
      p_context_->pt_outputs_.emplace_back(norm_adam_step->GetOutputs()[0]);
    }

    // weight_norm = p.data.norm()
    auto norm_wt = make_operator<NormOperator>(device_id, scalar_type);
    norm_wt->SetSynapseInput(p_context_->syn_inputs_[1 * num_params + i]);
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(2.0));
    norm_wt->AllocateAndAddSynapseNode(
        graph, stack, {output_metadata.at(i + 2)});
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(norm_wt->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(norm_wt->GetOutputs()[0]);

    // Note that these outputs are being filled just to keep GC
    // runtime happy No need to return these since updates on
    // weights, exp_avg, exp_avg_sq are all inplace
    synapse_helpers::tensor& syn_in_12 = mul_exp_avg->GetSynOutputs()[0];
    p_context_->syn_outputs_.emplace_back(syn_in_12);
    p_context_->pt_outputs_.emplace_back(mul_exp_avg->GetOutputs()[0]);

    synapse_helpers::tensor& syn_in_15 = add_exp_avg->GetSynOutputs()[0];
    p_context_->syn_outputs_.emplace_back(syn_in_15);
    p_context_->pt_outputs_.emplace_back(add_exp_avg->GetOutputs()[0]);

    synapse_helpers::tensor& syn_in_14 = mul_exp_avg_sq->GetSynOutputs()[0];
    p_context_->syn_outputs_.emplace_back(syn_in_14);
    p_context_->pt_outputs_.emplace_back(mul_exp_avg_sq->GetOutputs()[0]);

    synapse_helpers::tensor& syn_in_16 = addcmul_exp_avg_sq->GetSynOutputs()[0];
    p_context_->syn_outputs_.emplace_back(syn_in_16);
    p_context_->pt_outputs_.emplace_back(addcmul_exp_avg_sq->GetOutputs()[0]);
  }
}

static auto& OptimizerLambKernelRegistry = habana::KernelRegistry().add(
    "hpu::habanaOptimizerLambPhase1",
    KERNEL_FN(OptimizerLambPhase1Operator));
