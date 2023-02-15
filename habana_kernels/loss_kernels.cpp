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
#include "backend/synapse_helpers/layout_utils.h"
#include "backend/synapse_helpers/recipe.h"
#include "habana_device/HPUCheck.h"
#include "habana_device/hpu_cached_devices.h"
#include "habana_helpers/frontend_utils.h"
#include "habana_kernels/binary_kernels.h"
#include "habana_kernels/loss_kernels.h"
#include "habana_kernels/reduction_kernels.h"
#include "habana_kernels/tensor_shape_kernels.h"
#include "habana_kernels/threshold_kernels.h"
#include "habana_kernels/unary_kernels.h"
using namespace synapse_helpers::layouts;

using namespace torch;
using namespace habana;

std::vector<int64_t> KlDivOperator::compute_output_shape(
    const at::Tensor& self,
    int64_t reduction) {
  if (reduction == at::Reduction::Reduction::None) {
    return self.sizes().vec();
  } else {
    return {};
  }
}

void KlDivOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 4,
      "Incorrect size of inputs expected for kl_div operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isInt(), "Input type expected to be integer");
  TORCH_CHECK(inputs[3].isBool(), "Input type expected to be boolean");

  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[2].toInt();
  bool log_target = inputs[3].toBool();

  torch::jit::Stack stack;
  HabanaOperatorPtr log_exp_op;
  HabanaOperatorPtr threshold_op;
  if (log_target) {
    log_exp_op = static_cast<HabanaOperatorPtr>(
        make_operator<ExpOperator>(self.device().index(), self.scalar_type()));
    stack = {IValue(target)};
    log_exp_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    log_exp_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

  } else {
    log_exp_op = static_cast<HabanaOperatorPtr>(
        make_operator<LogOperator>(self.device().index(), self.scalar_type()));
    stack = {IValue(target)};
    log_exp_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    log_exp_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    threshold_op = make_operator<ThresholdBackwardOperator>(
        self.device().index(), self.scalar_type());
    threshold_op->SetSynapseInput(log_exp_op->GetSynOutputs()[0]);
    threshold_op->SetSynapseInput(p_context_->syn_inputs_[1]);
    stack = {IValue(log_exp_op->GetOutputs()[0]), IValue(target), IValue(0.0f)};
    threshold_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();
  }

  auto sub_op =
      make_operator<SubOperator>(self.device().index(), self.scalar_type());
  (log_target) ? sub_op->SetSynapseInput(p_context_->syn_inputs_[1])
               : sub_op->SetSynapseInput(threshold_op->GetSynOutputs()[0]);
  sub_op->SetSynapseInput(p_context_->syn_inputs_[0]);
  stack = {
      (log_target) ? IValue(target) : IValue(threshold_op->GetOutputs()[0]),
      IValue(self),
      IValue(1)};
  sub_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
  stack.clear();

  auto mul_op1 =
      make_operator<MulOperator>(self.device().index(), self.scalar_type());
  (log_target) ? mul_op1->SetSynapseInput(log_exp_op->GetSynOutputs()[0])
               : mul_op1->SetSynapseInput(p_context_->syn_inputs_[1]);
  mul_op1->SetSynapseInput(sub_op->GetSynOutputs()[0]);
  if (reduction == at::Reduction::Reduction::None) {
  }
  stack = {
      (log_target) ? IValue(log_exp_op->GetOutputs()[0]) : IValue(target),
      IValue(sub_op->GetOutputs()[0])};
  mul_op1->AllocateAndAddSynapseNode(
      graph,
      stack,
      (reduction == at::Reduction::Reduction::None) ? output_metadata
                                                    : OutputMetaDataVector(1));
  stack.clear();

  if (reduction != at::Reduction::Reduction::None) {
    auto sum_mean_op = (reduction == at::Reduction::Reduction::Sum)
        ? static_cast<HabanaOperatorPtr>(make_operator<SumOperator>(
              self.device().index(), self.scalar_type()))
        : static_cast<HabanaOperatorPtr>(make_operator<MeanOperator>(
              self.device().index(), self.scalar_type()));
    sum_mean_op->SetSynapseInput(mul_op1->GetSynOutputs()[0]);
    stack = {IValue(mul_op1->GetOutputs()[0]), IValue(self.scalar_type())};
    sum_mean_op->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(sum_mean_op->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(
        std::move(sum_mean_op->GetOutputs()[0]));
  } else {
    p_context_->syn_outputs_.emplace_back(
        std::move(mul_op1->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mul_op1->GetOutputs()[0]));
  }
}

OutputShapeInfRetType KlDivOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto self = inputs[0].toTensor();
  auto target = inputs[1].toTensor();
  int64_t reduction = inputs[2].toInt();
  bool log_target = inputs[3].toBool();

  torch::jit::Stack stack;
  HabanaOperatorPtr log_exp_op;
  HabanaOperatorPtr threshold_op;
  OutputShapeInfRetType out;
  OutputShapeInfRetType out_log_exp_op;
  OutputShapeInfRetType out_threshold_op;
  if (log_target) {
    log_exp_op =
        make_operator<ExpOperator>(self.device().index(), self.scalar_type());
    stack = {IValue(target)};
    out_log_exp_op = out.call_ComputeOutputShape(log_exp_op, stack);
    stack.clear();
  } else {
    log_exp_op =
        make_operator<LogOperator>(self.device().index(), self.scalar_type());
    stack = {IValue(target)};
    out_log_exp_op = out.call_ComputeOutputShape(log_exp_op, stack);
    stack.clear();

    threshold_op = make_operator<ThresholdBackwardOperator>(
        self.device().index(), self.scalar_type());
    stack = {
        IValue(std::get<1>(out_log_exp_op.GetOutputTensor(0))),
        IValue(target),
        IValue(0.0f)};
    out_threshold_op = out.call_ComputeOutputShape(threshold_op, stack);
    stack.clear();
  }

  auto sub_op =
      make_operator<SubOperator>(self.device().index(), self.scalar_type());
  stack = {
      (log_target) ? IValue(target)
                   : IValue(std::get<1>(out_threshold_op.GetOutputTensor(0))),
      IValue(self),
      IValue(1)};
  auto out_sub_op = out.call_ComputeOutputShape(sub_op, stack);
  stack.clear();

  auto mul_op1 =
      make_operator<MulOperator>(self.device().index(), self.scalar_type());
  stack = {
      (log_target) ? IValue(std::get<1>(out_log_exp_op.GetOutputTensor(0)))
                   : IValue(target),
      IValue(std::get<1>(out_sub_op.GetOutputTensor(0)))};
  auto out_mul_op1 = out.call_ComputeOutputShape(mul_op1, stack);
  stack.clear();

  if (reduction != at::Reduction::Reduction::None) {
    auto sum_mean_op = (reduction == at::Reduction::Reduction::Sum)
        ? static_cast<HabanaOperatorPtr>(make_operator<SumOperator>(
              self.device().index(), self.scalar_type()))
        : static_cast<HabanaOperatorPtr>(make_operator<MeanOperator>(
              self.device().index(), self.scalar_type()));
    stack = {
        IValue(std::get<1>(out_mul_op1.GetOutputTensor(0))),
        IValue(self.scalar_type())};
    auto out_sum_mean_op = out.call_ComputeOutputShape(sum_mean_op, stack);
    stack.clear();

    auto out_tensor = out_sum_mean_op.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else {
    auto out_tensor = out_mul_op1.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  }
  return out;
}

void KlDivBwdOperator::AllocateAndAddSynapseNode(
    synapse_helpers::graph& graph,
    torch::jit::Stack& inputs,
    const OutputMetaDataVector& output_metadata) {
  TORCH_CHECK(
      inputs.size() == 5,
      "Incorrect size of inputs expected for kl_div backward operator");
  TORCH_CHECK(inputs[0].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[1].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[2].isTensor(), "Input type expected to be tensor");
  TORCH_CHECK(inputs[3].isInt(), "Input type expected to be integer");
  TORCH_CHECK(inputs[4].isBool(), "Input type expected to be boolean");

  auto grad_out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto target = inputs[2].toTensor();
  int64_t reduction = inputs[3].toInt();
  bool log_target = inputs[4].toBool();

  torch::jit::Stack stack;
  HabanaOperatorPtr exp_op;
  if (log_target) {
    exp_op = static_cast<HabanaOperatorPtr>(
        make_operator<ExpOperator>(self.device().index(), self.scalar_type()));
    stack = {IValue(target)};
    exp_op->SetSynapseInput(p_context_->syn_inputs_[2]);
    exp_op->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();
  }

  if (reduction != at::Reduction::Reduction::None) {
    auto shape = self.sizes().vec();
    int flattened_size = std::accumulate(
        shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
    int64_t data[1];
    data[0] = flattened_size;
    IntArrayRef size_arr(data, 1);
    auto sum_mean_op = (reduction == at::Reduction::Reduction::Sum)
        ? static_cast<HabanaOperatorPtr>(make_operator<ReduceSumBwdOperator>(
              self.device().index(), self.scalar_type()))
        : static_cast<HabanaOperatorPtr>(make_operator<ReduceMeanBwdOperator>(
              self.device().index(), self.scalar_type()));
    sum_mean_op->SetSynapseInput(p_context_->syn_inputs_[0]);
    stack = {IValue(grad_out), IValue(size_arr), IValue(0)};
    sum_mean_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto shape1 = self.sizes().vec();
    auto reshape_op = make_operator<ReshapeOperator>(
        self.device().index(), self.scalar_type());
    reshape_op->SetSynapseInput(sum_mean_op->GetSynOutputs()[0]);
    stack = {IValue(sum_mean_op->GetOutputs()[0]), IValue(shape1)};
    reshape_op->AllocateAndAddSynapseNode(
        graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mul_op1 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op1->SetSynapseInput(reshape_op->GetSynOutputs()[0]);
    (log_target) ? mul_op1->SetSynapseInput(exp_op->GetSynOutputs()[0])
                 : mul_op1->SetSynapseInput(p_context_->syn_inputs_[2]);
    stack = {
        IValue(reshape_op->GetOutputs()[0]),
        (log_target) ? IValue(exp_op->GetOutputs()[0]) : IValue(target)};
    mul_op1->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mul_op3 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op3->SetSynapseInput(mul_op1->GetSynOutputs()[0]);
    stack = {IValue(mul_op1->GetOutputs()[0]), IValue(-1)};
    mul_op3->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(mul_op3->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mul_op3->GetOutputs()[0]));
  } else {
    auto mul_op2 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op2->SetSynapseInput(p_context_->syn_inputs_[0]);
    (log_target) ? mul_op2->SetSynapseInput(exp_op->GetSynOutputs()[0])
                 : mul_op2->SetSynapseInput(p_context_->syn_inputs_[2]);
    stack = {
        IValue(grad_out),
        (log_target) ? IValue(exp_op->GetOutputs()[0]) : IValue(target)};
    mul_op2->AllocateAndAddSynapseNode(graph, stack, OutputMetaDataVector(1));
    stack.clear();

    auto mul_op4 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    mul_op4->SetSynapseInput(mul_op2->GetSynOutputs()[0]);
    stack = {IValue(mul_op2->GetOutputs()[0]), IValue(-1)};
    mul_op4->AllocateAndAddSynapseNode(graph, stack, output_metadata);
    stack.clear();

    p_context_->syn_outputs_.emplace_back(
        std::move(mul_op4->GetSynOutputs()[0]));
    p_context_->pt_outputs_.emplace_back(std::move(mul_op4->GetOutputs()[0]));
  }
}

OutputShapeInfRetType KlDivBwdOperator::ComputeOutputShape(
    torch::jit::Stack& inputs) {
  auto grad_out = inputs[0].toTensor();
  auto self = inputs[1].toTensor();
  auto target = inputs[2].toTensor();
  int64_t reduction = inputs[3].toInt();
  bool log_target = inputs[4].toBool();

  torch::jit::Stack stack;
  HabanaOperatorPtr exp_op;
  OutputShapeInfRetType out;
  OutputShapeInfRetType out_exp_op;
  if (log_target) {
    exp_op =
        make_operator<ExpOperator>(self.device().index(), self.scalar_type());
    stack = {IValue(target)};
    out_exp_op = out.call_ComputeOutputShape(exp_op, stack);
    stack.clear();
  }

  if (reduction != at::Reduction::Reduction::None) {
    auto shape = self.sizes().vec();
    int flattened_size = std::accumulate(
        shape.cbegin(), shape.cend(), 1, std::multiplies<int>());
    int64_t data[1];
    data[0] = flattened_size;
    IntArrayRef size_arr(data, 1);
    auto sum_mean_op = (reduction == at::Reduction::Reduction::Sum)
        ? static_cast<HabanaOperatorPtr>(make_operator<ReduceSumBwdOperator>(
              self.device().index(), self.scalar_type()))
        : static_cast<HabanaOperatorPtr>(make_operator<ReduceMeanBwdOperator>(
              self.device().index(), self.scalar_type()));
    stack = {IValue(grad_out), IValue(size_arr), IValue(0)};
    auto out_sum_mean_op = out.call_ComputeOutputShape(sum_mean_op, stack);
    stack.clear();

    auto shape1 = self.sizes().vec();
    auto reshape_op = make_operator<ReshapeOperator>(
        self.device().index(), self.scalar_type());
    stack = {
        IValue(std::get<1>(out_sum_mean_op.GetOutputTensor(0))),
        IValue(shape1)};
    auto out_reshape_op = out.call_ComputeOutputShape(reshape_op, stack);
    stack.clear();

    auto mul_op1 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());

    stack = {
        IValue(std::get<1>(out_reshape_op.GetOutputTensor(0))),
        (log_target) ? IValue(std::get<1>(out_exp_op.GetOutputTensor(0)))
                     : IValue(target)};
    auto out_mul_op1 = out.call_ComputeOutputShape(mul_op1, stack);
    stack.clear();

    auto mul_op3 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    stack = {IValue(std::get<1>(out_mul_op1.GetOutputTensor(0))), IValue(-1)};
    auto out_mul_op3 = out.call_ComputeOutputShape(mul_op3, stack);
    stack.clear();

    auto out_tensor = out_mul_op3.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  } else {
    auto mul_op2 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    stack = {
        IValue(grad_out),
        (log_target) ? IValue(std::get<1>(out_exp_op.GetOutputTensor(0)))
                     : IValue(target)};
    auto out_mul_op2 = out.call_ComputeOutputShape(mul_op2, stack);
    stack.clear();

    auto mul_op4 =
        make_operator<MulOperator>(self.device().index(), self.scalar_type());
    stack = {IValue(std::get<1>(out_mul_op2.GetOutputTensor(0))), IValue(-1)};
    auto out_mul_op4 = out.call_ComputeOutputShape(mul_op4, stack);
    stack.clear();

    auto out_tensor = out_mul_op4.GetOutputTensor(0);
    out.MoveToOutput(std::move(out_tensor));
  }
  return out;
}

static auto& LossKernelsKernelRegistry =
    habana::KernelRegistry()
        .add("aten::kl_div", KERNEL_FN(KlDivOperator))
        .add("aten::kl_div_backward", KERNEL_FN(KlDivBwdOperator));
