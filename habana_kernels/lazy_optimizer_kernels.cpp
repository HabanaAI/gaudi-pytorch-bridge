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

#include "lazy_optimizer_kernels.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ops/optimizer.h"
#include "habana_lazy/view_utils.h"

using namespace at;
using namespace habana;

namespace habana_lazy {

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_sgd_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor,
    float mom,
    bool nesterov) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  HbLazyTensor::StepMarker({});
  LazyOp<::std::tuple<at::Tensor&, at::Tensor&>> k{
      "hpu::habanaOptimizerSparseSgd",
      {gradients,
       weights_in,
       moments_in,
       indices,
       learning_rate,
       valid_count_tensor,
       mom,
       nesterov},
      {weights_in.sizes().vec(), moments_in.sizes().vec()}};

  auto result =
      k.call(::std::tuple<at::Tensor&, at::Tensor&>(weights_in, moments_in));

  flush_op();

  return result;
}

std::tuple<torch::Tensor&, torch::Tensor&>
optimizer_sparse_adagrad_with_valid_count_hpu_lazy(
    const Tensor& gradients,
    Tensor& weights_in,
    Tensor& moments_in,
    const Tensor& indices,
    const Tensor& learning_rate,
    const Tensor& valid_count_tensor) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  LazyOp<::std::tuple<at::Tensor&, at::Tensor&>> k{
      "hpu::habanaOptimizerSparseAdagrad",
      {gradients,
       weights_in,
       moments_in,
       indices,
       learning_rate,
       valid_count_tensor},
      {weights_in.sizes().vec(), moments_in.sizes().vec()}};

  return k.call(::std::tuple<at::Tensor&, at::Tensor&>(weights_in, moments_in));
}

void optimizer_adamw_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& exp_avg,
    TensorList& exp_avg_sq,
    at::Tensor& lr_t,
    at::Tensor& neg_step_t,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float modified_wd) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> gradients_v;
  std::vector<at::Tensor> weights_v;
  std::vector<at::Tensor> exp_avg_v;
  std::vector<at::Tensor> exp_avg_sq_v;

  std::copy(
      gradients.begin(), gradients.end(), std::back_inserter(gradients_v));
  std::copy(weights.begin(), weights.end(), std::back_inserter(weights_v));
  std::copy(exp_avg.begin(), exp_avg.end(), std::back_inserter(exp_avg_v));
  std::copy(
      exp_avg_sq.begin(), exp_avg_sq.end(), std::back_inserter(exp_avg_sq_v));

  handle_collective(gradients_v);
  handle_collective(weights_v);
  handle_collective(exp_avg_v);
  handle_collective(exp_avg_sq_v);
  at::Tensor modified_wd_t = get_tensor_for_scalar(modified_wd);

  bool is_wd_modified = modified_wd != 1.0;
  auto func = [gradients_v = std::move(gradients_v),
               weights_v = std::move(weights_v),
               exp_avg_v = std::move(exp_avg_v),
               exp_avg_sq_v = std::move(exp_avg_sq_v),
               lr_t,
               neg_step_t,
               beta1,
               beta2,
               epsilon,
               modified_wd_t,
               is_wd_modified]() mutable {
    TensorList gradients = gradients_v;
    TensorList weights = weights_v;
    TensorList exp_avg = exp_avg_v;
    TensorList exp_avg_sq = exp_avg_sq_v;

    auto hl_lr_t = GetHbLazyTensor(lr_t);
    auto hl_neg_step_t = GetHbLazyTensor(neg_step_t);
    auto hl_modified_wd_t = GetHbLazyTensor(modified_wd_t);

    ir::NodePtr node = std::make_shared<ir::OptimizerFusedAdamw>(
        gradients,
        weights,
        exp_avg,
        exp_avg_sq,
        lr_t,
        neg_step_t,
        beta1,
        beta2,
        epsilon,
        modified_wd_t,
        is_wd_modified);

    int64_t out_index = 0;

    auto hlweight = habana_lazy::GetHbLazyTensor(weights[0]);
    node->set_as_output_tensor_list();
    habana_lazy::ir::Value& out = hlweight.IrSetNode(node);

    habana_lazy::ir::NodePtr node_unpack =
        std::make_shared<habana_lazy::ir::ListUnpack>(out);

    for (size_t i = 0; i < weights.size(); i++) {
      if (is_wd_modified) {
        auto hl_wd = GetHbLazyTensor(weights[i]);
        hl_wd.IrSetNode(node_unpack, out_index++);
      }

      auto hl_exp_avg = GetHbLazyTensor(exp_avg[i]);
      hl_exp_avg.IrSetNode(node_unpack, out_index++);

      auto hl_exp_avg_1 = GetHbLazyTensor(exp_avg[i]);
      hl_exp_avg_1.IrSetNode(node_unpack, out_index++);

      auto hl_exp_avg_sq = GetHbLazyTensor(exp_avg_sq[i]);
      hl_exp_avg_sq.IrSetNode(node_unpack, out_index++);

      auto hl_exp_avg_sq_1 = GetHbLazyTensor(exp_avg_sq[i]);
      hl_exp_avg_sq_1.IrSetNode(node_unpack, out_index++);

      HbLazyTensorViews::CustomKernelAddNodeInplace(
          weights[i], node_unpack, out_index);
    }

    flush_op();
  };
  auto vector_of_inputs = std::vector<c10::IValue>{
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      lr_t,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      modified_wd_t,
      is_wd_modified};
  RUNNING_HASH_COMBINE_OPERATOR(hpu::habanaOptimizerAdamW, vector_of_inputs);
  RUN_MANUAL_OP_NO_RETURN_WITH_ACC_THREAD(optimizer_adamw, func)
}

std::tuple<
    std::vector<at::Tensor>,
    std::vector<at::Tensor>,
    std::vector<at::Tensor>>
optimizer_lamb_phase1_hpu_lazy(
    const std::vector<at::Tensor>& gradients,
    std::vector<at::Tensor>& weights,
    std::vector<at::Tensor>& exp_avg,
    std::vector<at::Tensor>& exp_avg_sq,
    const at::Tensor& clip_global_grad_norm,
    const int grad_averaging,
    const float lr,
    const float beta1,
    const float beta2,
    const float epsilon,
    const int step,
    const int bias_correction,
    const float weight_decay) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;
  static_cast<void>(lr);

  auto hl_clip_global = GetHbLazyTensor(clip_global_grad_norm);
  updateDstDependencies(clip_global_grad_norm);
  // TODO: SW-69618 JIT optimization passes are failing for
  // habanaOptimizerLambPhase1 and habanaOptimizerLambPhase2 because we
  // dont support tensorlist in lowering that matches kernel schema.
  // Adding unpack will return TensorList, which is not supported as
  // graph output.
  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  float bias_correction1 = 1.0, bias_correction2 = 1.0;
  if (bias_correction) {
    bias_correction1 = 1.0 - std::pow(beta1, step);
    bias_correction2 = 1.0 - std::pow(beta2, step);
  }

  float beta3 = 1.0;
  if (grad_averaging) {
    beta3 = 1 - beta1;
  }

  auto bias_correction1_t = get_tensor_for_scalar(bias_correction1);
  auto bias_correction2_t = get_tensor_for_scalar(bias_correction2);

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedLambPhase1>(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      clip_global_grad_norm,
      beta1,
      beta2,
      beta3,
      epsilon,
      bias_correction1_t,
      bias_correction2_t,
      weight_decay);

  int64_t out_index = 0;

  auto context = habana_lazy_executor.getDeviceExecutionContext(0);

  std::vector<Tensor> weight_norm_vec, adam_norm_vec, adam_step_vec;
  for (size_t i = 0; i < weights.size(); i++) {
    auto adam_step = empty_hpu_lazy(
        weights[i].sizes(),
        weights[i].options(),
        weights[i].suggest_memory_format(),
        false);
    auto hl_adam_step = GetHbLazyTensor(adam_step);
    hl_adam_step.IrSetNode(node, out_index++);

    context->m_retained_tensor_list.emplace_back(adam_step);
    adam_step_vec.push_back(adam_step);

    auto adam_norm = empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_adam_norm = GetHbLazyTensor(adam_norm);
    hl_adam_norm.IrSetNode(node, out_index++);

    context->m_retained_tensor_list.emplace_back(adam_norm);
    adam_norm_vec.push_back(adam_norm);

    auto weight_norm = empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_weight_norm = GetHbLazyTensor(weight_norm);
    hl_weight_norm.IrSetNode(node, out_index++);

    context->m_retained_tensor_list.emplace_back(weight_norm);
    weight_norm_vec.push_back(weight_norm);

    // add the tensors that are updated inplace
    auto exp_avg_temp = empty_hpu_lazy(
        exp_avg[i].sizes(),
        exp_avg[i].options(),
        exp_avg[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_temp = GetHbLazyTensor(exp_avg_temp);
    hl_exp_avg_temp.IrSetNode(node, out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_temp);

    auto hl_exp_avg = GetHbLazyTensor(exp_avg[i]);
    hl_exp_avg.IrSetNode(node, out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg[i]);

    auto exp_avg_sq_temp = empty_hpu_lazy(
        exp_avg_sq[i].sizes(),
        exp_avg_sq[i].options(),
        exp_avg_sq[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_sq_temp = GetHbLazyTensor(exp_avg_sq_temp);
    hl_exp_avg_sq_temp.IrSetNode(node, out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq_temp);

    auto hl_exp_avg_sq = GetHbLazyTensor(exp_avg_sq[i]);
    hl_exp_avg_sq.IrSetNode(node, out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq[i]);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }

  flush_op();
  return std::tie(weight_norm_vec, adam_norm_vec, adam_step_vec);
}

void optimizer_adagrad_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  // Refer comment on SW-69618 in this file
  exec::OptPassCfg::GetInstance()->BkupAndDisableAndAllOptPass();

  LazyOptimizationOp<void> loo(
      "hpu::habanaOptimizerFusedAdagrad",
      {gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon});

  loo.call(weights, variances, ADAGRAD);
}

void optimizer_sgd_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  LazyOptimizationOp<void> loo(
      "hpu::habanaOptimizerFusedSGD",
      {gradients, weights, lr, wd, mom, damp, nesterov});

  loo.call(weights);
}

void optimizer_sgd_momentum_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const at::Tensor& mom,
    const float wd,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  LazyOptimizationOp<void> loo(
      "hpu::habanaOptimizerFusedSGDMomentum",
      {gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov});
  loo.call(weights, momentum, OPTIMIZER::SGD_MOMENTUM);
}

} // namespace habana_lazy
