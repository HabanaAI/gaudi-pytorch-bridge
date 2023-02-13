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
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/tensor_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_kernels/binary_inplace_kernels.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/compare_kernels.h"
#include "habana_kernels/norm_kernels.h"
#include "habana_kernels/optimizer_lamb.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/unary_kernels.h"
#include "simple_generic_kernel.h"

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

void OptimizerLambPhase2Operator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 7,
      "Incorrect size of inputs for lamb optimizer ph2 graph creation call");

  auto weights = inputs[0].toTensorList();
  auto adam_norm = inputs[1].toTensorList();
  auto weight_norm = inputs[2].toTensorList();
  auto adam_step = inputs[3].toTensorList();
  auto nstep = inputs[4].toTensor();
  auto weight_decay = inputs[5].toScalar();
  auto use_lamb = inputs[6].toScalar();

  auto device_id = weights.get(0).device().index();
  auto scalar_type = weights.get(0).scalar_type();
  auto num_params = static_cast<int>(weights.size());
  torch::jit::Stack stack;
  for (auto i = 0; i < num_params; i++) {
    // Synapse Graph for single parameter update to be created here
    // All synapse input tensor references are there in a single std::vector
    // arranged as follows,
    // weights ; adam_norm ; weight_norm; adam_step; trust_ratio; nstep

    auto mul4 = make_operator<MulOperator>(device_id, scalar_type);
    if ((weight_decay.toFloat() != 0.0) || use_lamb.toInt()) {
      // weight_norm / adam_norm
      auto div_lp = make_operator<DivOperator>(device_id, scalar_type);
      div_lp->SetSynapseInput(p_context_->syn_inputs_[2 * num_params + i]);
      div_lp->SetSynapseInput(p_context_->syn_inputs_[num_params + i]);
      stack.emplace_back(IValue(weight_norm.get(i)));
      stack.emplace_back(IValue(adam_norm.get(i)));
      div_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      // weight_norm + adam_norm
      auto add1_lp = make_operator<AddOperator>(device_id, scalar_type);
      add1_lp->SetSynapseInput(p_context_->syn_inputs_[2 * num_params + i]);
      add1_lp->SetSynapseInput(p_context_->syn_inputs_[num_params + i]);
      stack.emplace_back(IValue(weight_norm.get(i)));
      stack.emplace_back(IValue(adam_norm.get(i)));
      stack.emplace_back(IValue(1.0));
      add1_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      // mask = (weight_norm + adam_norm == 0)
      auto eq1_lp = make_operator<EqOperator>(device_id, scalar_type);
      eq1_lp->SetSynapseInput(add1_lp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(add1_lp->GetOutputs()[0]));
      stack.emplace_back(IValue(0.0));
      eq1_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      std::string node_type = "cast_i8_to_f32";
      auto cast1 = make_operator<CastOperator>(device_id, node_type);
      cast1->SetSynapseInput(eq1_lp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(eq1_lp->GetOutputs()[0]));
      stack.emplace_back(IValue(c10::ScalarType::Float));
      auto md = OutputMetaDataVector(1);
      md[0].dtype = stack[1].toScalarType();
      cast1->AllocateAndAddSynapseNode(graph, stack, md);
      stack.clear();

      // mul1 = mask [* trust_ratio(=1)]
      auto& mul1 = cast1;

      // imask = (mask == 0)
      auto eq2_lp = make_operator<EqOperator>(device_id, scalar_type);
      eq2_lp->SetSynapseInput(cast1->GetSynOutputs()[0]);
      stack.emplace_back(IValue(cast1->GetOutputs()[0]));
      stack.emplace_back(IValue(0));
      eq2_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      node_type = "cast_i8_to_f32";
      auto cast2 = make_operator<CastOperator>(device_id, node_type);
      cast2->SetSynapseInput(eq2_lp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(eq2_lp->GetOutputs()[0]));
      stack.emplace_back(IValue(c10::ScalarType::Float));
      md[0].dtype = stack[1].toScalarType();
      cast2->AllocateAndAddSynapseNode(graph, stack, md);
      stack.clear();

      // mul2 = imask * weight_norm / adam_norm
      auto mul2 = make_operator<MulOperator>(device_id, scalar_type);
      mul2->SetSynapseInput(cast2->GetSynOutputs()[0]);
      mul2->SetSynapseInput(div_lp->GetSynOutputs()[0]);
      stack.emplace_back(IValue(cast2->GetOutputs()[0]));
      stack.emplace_back(IValue(div_lp->GetOutputs()[0]));
      mul2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      // trust_ratio = mask * trust_ratio(=1) + imask * weight_norm / adam_norm
      auto add2_lp = make_operator<AddOperator>(device_id, scalar_type);
      add2_lp->SetSynapseInput(mul1->GetSynOutputs()[0]);
      add2_lp->SetSynapseInput(mul2->GetSynOutputs()[0]);
      stack.emplace_back(IValue(mul1->GetOutputs()[0]));
      stack.emplace_back(IValue(mul2->GetOutputs()[0]));
      stack.emplace_back(IValue(1.0));
      add2_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      // trust_ratio * -step
      auto mul3 = make_operator<MulOperator>(device_id, scalar_type);
      mul3->SetSynapseInput(add2_lp->GetSynOutputs()[0]);
      mul3->SetSynapseInput(p_context_->syn_inputs_[4 * num_params]);
      stack.emplace_back(IValue(add2_lp->GetOutputs()[0]));
      stack.emplace_back(IValue(nstep));
      mul3->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

      // trust_ratio * -step * adam_step
      mul4->SetSynapseInput(mul3->GetSynOutputs()[0]);
      mul4->SetSynapseInput(p_context_->syn_inputs_[3 * num_params + i]);
      stack.emplace_back(IValue(mul3->GetOutputs()[0]));
      stack.emplace_back(IValue(adam_step.get(i)));
      mul4->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();

    } else {
      // -step * adam_step
      mul4->SetSynapseInput(p_context_->syn_inputs_[3 * num_params + i]);
      mul4->SetSynapseInput(p_context_->syn_inputs_[4 * num_params]);
      stack.emplace_back(IValue(adam_step.get(i)));
      stack.emplace_back(IValue(nstep));
      mul4->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
      stack.clear();
    }

    // p.data.add_(adam_step)
    auto add3_lp = make_operator<AddInplaceOperator>(device_id, scalar_type);
    add3_lp->SetSynapseInput(p_context_->syn_inputs_[i]);
    add3_lp->SetSynapseInput(mul4->GetSynOutputs()[0]);
    stack.emplace_back(IValue(weights.get(i)));
    stack.emplace_back(IValue(mul4->GetOutputs()[0]));
    stack.emplace_back(IValue(1.0));
    add3_lp->AllocateAndAddSynapseNode(graph, stack, {output_metadata.at(i)});
    stack.clear();

    // Note that these outputs are being filled just to keep GC
    // runtime happy No need to return these since updates on
    // weights are all inplace
    p_context_->syn_outputs_.emplace_back(
        std::move(add3_lp->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(add3_lp->GetOutputs()[0]);
  }
}

void OptNormFusedNormOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
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
  auto cat_grads = make_operator<CatOperator>(device_id, scalar_type);
  std::vector<int64_t> shape{1, 1};
  for (auto i = 0; i < num_params; i++) {
    // add node to compute reduce_sum_square
    auto sum_lp = make_operator<SumSquareOperator>(device_id, scalar_type);
    sum_lp->SetSynapseInput(p_context_->syn_inputs_[i]);
    stack.emplace_back(IValue(gradients.get(i)));
    stack.emplace_back(IValue(scalar_type));
    sum_lp->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    // Unsqueeze sum output (tensor of shape {1}) to new tensor of shape
    // {1,1} in prep for cat
    auto reshape_lp = make_operator<ReshapeOperator>(device_id, scalar_type);
    reshape_lp->SetSynapseInput(sum_lp->GetSynOutputs()[0]);
    stack.emplace_back(IValue(sum_lp->GetOutputs()[0]));
    stack.emplace_back(IValue(shape));
    reshape_lp->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    // each unsqueezed grad_norm connected to cat node
    cat_input.push_back(reshape_lp->GetOutputs()[0]);
    cat_grads->SetSynapseInput(reshape_lp->GetSynOutputs()[0]);
    stack.clear();
  }

  // grads are concatened into a single big tensor of shape
  // {num_params,1}
  stack.emplace_back(IValue(cat_input));
  stack.emplace_back(IValue(0));
  cat_grads->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // add node to compute reduce_sum
  auto sum_final = make_operator<SumOperator>(device_id, scalar_type);
  sum_final->SetSynapseInput(cat_grads->GetSynOutputs()[0]);
  stack.emplace_back(IValue(cat_grads->GetOutputs()[0]));
  stack.emplace_back(IValue(scalar_type));
  sum_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // global_grad_norm = global_grad_norm.sqrt()
  auto sqrt_final = make_operator<SqrtOperator>(device_id, scalar_type);
  sqrt_final->SetSynapseInput(sum_final->GetSynOutputs()[0]);
  stack.emplace_back(IValue(sum_final->GetOutputs()[0]));
  sqrt_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  std::shared_ptr<HabanaOperator> global_grad_norm = sqrt_final;

  // if global_grad_norm > max_grad_norm:
  //    clip_global_grad_norm = global_grad_norm / max_grad_norm
  // else:
  //    clip_global_grad_norm = 1.0

  if (habana_helpers::pytorch_to_synapse_type(max_grad_norm.type()) !=
      habana_helpers::pytorch_to_synapse_type(scalar_type)) {
    auto cast00 = make_operator<CastOperator>(
        device_id,
        "cast_" + habana_helpers::name_suffix_from_type(scalar_type) + "_to_" +
            habana_helpers::name_suffix_from_type(max_grad_norm.type()));

    cast00->SetSynapseInput(global_grad_norm->GetSynOutputs()[0]);
    stack.emplace_back(IValue(global_grad_norm->GetOutputs()[0]));

    // This is on purpose as not all pytorch types have their synapse
    // counterparts
    scalar_type = habana_helpers::synapse_to_pytorch_type(
        habana_helpers::pytorch_to_synapse_type(max_grad_norm.type()));

    stack.emplace_back(IValue(scalar_type));
    auto md = OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    cast00->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    global_grad_norm = cast00;
  }

  // global_grad_norm / max_grad_norm
  auto div_final = make_operator<DivOperator>(device_id, scalar_type);
  div_final->SetSynapseInput(global_grad_norm->GetSynOutputs()[0]);
  stack.emplace_back(IValue(global_grad_norm->GetOutputs()[0]));
  stack.emplace_back(IValue(max_grad_norm));
  div_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // mask = global_grad_norm < max_grad_norm
  auto lt_final = make_operator<LtOperator>(device_id, scalar_type);
  lt_final->SetSynapseInput(global_grad_norm->GetSynOutputs()[0]);
  stack.emplace_back(IValue(global_grad_norm->GetOutputs()[0]));
  stack.emplace_back(IValue(max_grad_norm));
  lt_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  const std::string node_type =
      "cast_i8_to_" + habana_helpers::name_suffix_from_type(scalar_type);
  auto cast1 = make_operator<CastOperator>(device_id, node_type);
  cast1->SetSynapseInput(lt_final->GetSynOutputs()[0]);
  stack.emplace_back(IValue(lt_final->GetOutputs()[0]));
  stack.emplace_back(IValue(scalar_type));
  auto md = OutputMetaDataVector(1);
  md[0].dtype = stack[1].toScalarType();
  cast1->AllocateAndAddSynapseNode(graph, stack, md);
  stack.clear();

  std::shared_ptr<MulOperator> mul1;

  if (habana_helpers::pytorch_to_synapse_type(clip_norm.scalar_type()) !=
      habana_helpers::pytorch_to_synapse_type(scalar_type)) {
    auto cast01 = make_operator<CastOperator>(
        device_id,
        "cast_" +
            habana_helpers::name_suffix_from_type(clip_norm.scalar_type()) +
            "_to_" + habana_helpers::name_suffix_from_type(scalar_type));
    cast01->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    stack.emplace_back(IValue(clip_norm));
    stack.emplace_back(IValue(scalar_type));
    auto md = OutputMetaDataVector(1);
    md[0].dtype = stack[1].toScalarType();
    cast01->AllocateAndAddSynapseNode(graph, stack, md);
    stack.clear();

    // mul1 = mask * cast(clip_norm(=1))
    mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(cast1->GetSynOutputs()[0]);
    mul1->SetSynapseInput(cast01->GetSynOutputs()[0]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(cast01->GetOutputs()[0]));
    mul1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
  } else {
    // mul1 = mask * clip_norm(=1)
    mul1 = make_operator<MulOperator>(device_id, scalar_type);
    mul1->SetSynapseInput(cast1->GetSynOutputs()[0]);
    mul1->SetSynapseInput(p_context_->syn_inputs_[num_params]);
    stack.emplace_back(IValue(cast1->GetOutputs()[0]));
    stack.emplace_back(IValue(clip_norm));
    mul1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
  }

  // imask = (mask == 0)
  auto eq_final = make_operator<EqOperator>(device_id, scalar_type);
  eq_final->SetSynapseInput(cast1->GetSynOutputs()[0]);
  stack.emplace_back(IValue(cast1->GetOutputs()[0]));
  stack.emplace_back(IValue(0));
  eq_final->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  auto cast2 = make_operator<CastOperator>(device_id, node_type);
  cast2->SetSynapseInput(eq_final->GetSynOutputs()[0]);
  stack.emplace_back(IValue(eq_final->GetOutputs()[0]));
  stack.emplace_back(IValue(scalar_type));
  md[0].dtype = stack[1].toScalarType();
  cast2->AllocateAndAddSynapseNode(graph, stack, md);
  stack.clear();

  // mul2 = imask * (global_grad_norm / max_grad_norm)
  auto mul2 = make_operator<MulOperator>(device_id, scalar_type);
  mul2->SetSynapseInput(cast2->GetSynOutputs()[0]);
  mul2->SetSynapseInput(div_final->GetSynOutputs()[0]);
  stack.emplace_back(IValue(cast2->GetOutputs()[0]));
  stack.emplace_back(IValue(div_final->GetOutputs()[0]));
  mul2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  // out = mask * clip_norm(=1) + imask * (global_grad_norm / max_grad_norm)
  auto add = make_operator<AddOperator>(device_id, scalar_type);
  add->SetSynapseInput(mul1->GetSynOutputs()[0]);
  add->SetSynapseInput(mul2->GetSynOutputs()[0]);
  stack.emplace_back(IValue(mul1->GetOutputs()[0]));
  stack.emplace_back(IValue(mul2->GetOutputs()[0]));
  stack.emplace_back(IValue(1.0));
  add->AllocateAndAddSynapseNode(graph, stack, output_metadata);
  stack.clear();

  p_context_->syn_outputs_.emplace_back(std::move(add->GetSynOutputs()[0]));
  p_context_->pt_outputs_.emplace_back(add->GetOutputs()[0]);
}

static auto& OptimizerLambKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habanaOptimizerLambFusedNorm",
            KERNEL_FN(OptNormFusedNormOperator))
        .add(
            "hpu::habanaOptimizerLambPhase1",
            KERNEL_FN(OptimizerLambPhase1Operator))
        .add(
            "hpu::habanaOptimizerLambPhase2",
            KERNEL_FN(OptimizerLambPhase2Operator));
