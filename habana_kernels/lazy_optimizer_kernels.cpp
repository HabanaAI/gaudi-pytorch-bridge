/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "lazy_optimizer_kernels.h"
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
      {},
      {weights_in.sizes().vec(), moments_in.sizes().vec()}};

  auto result =
      k.call(::std::tuple<at::Tensor&, at::Tensor&>(weights_in, moments_in));

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }
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

  LazyOp<::std::tuple<at::Tensor&, at::Tensor&>> k{
      "hpu::habanaOptimizerSparseAdagrad",
      {gradients,
       weights_in,
       moments_in,
       indices,
       learning_rate,
       valid_count_tensor},
      {},
      {weights_in.sizes().vec(), moments_in.sizes().vec()}};

  return k.call(::std::tuple<at::Tensor&, at::Tensor&>(weights_in, moments_in));
}

void optimizer_ema_hpu_lazy(
    const at::TensorList& model_inputs,
    at::TensorList& updated_ema,
    const at::Tensor& decay) {
  PT_LAZY_TRACE;

  ir::NodePtr node =
      std::make_shared<ir::OptimizerFusedEMA>(model_inputs, updated_ema, decay);

  int64_t out_index = 0;

  auto hl_ema = GetHbLazyTensor(updated_ema[0]);
  ir::Value& out = hl_ema.CurrentIrValue();
  node->set_as_output_tensor_list();
  out.SetNode(
      node, hl_ema.GetDevice(), hl_ema.GetSizes(), hl_ema.dtype_optional());

  ir::NodePtr node_unpack = std::make_shared<ir::ListUnpack>(out);

  for (size_t i = 0; i < updated_ema.size(); i++) {
    HbLazyTensorViews::CustomKernelAddNodeInplace(
        updated_ema[i], node_unpack, out_index);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }
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

  auto hl_lr_t = GetHbLazyTensor(lr_t);
  auto hl_neg_step_t = GetHbLazyTensor(neg_step_t);

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
      modified_wd);

  int64_t out_index = 0;

  auto hlweight = habana_lazy::GetHbLazyTensor(weights[0]);
  habana_lazy::ir::Value& out = hlweight.CurrentIrValue();
  node->set_as_output_tensor_list();
  out.SetNode(
      node,
      hlweight.GetDevice(),
      hlweight.GetSizes(),
      hlweight.dtype_optional());

  habana_lazy::ir::NodePtr node_unpack =
      std::make_shared<habana_lazy::ir::ListUnpack>(out);

  for (size_t i = 0; i < weights.size(); i++) {
    if (modified_wd != 1.0) {
      auto hl_wd = GetHbLazyTensor(weights[i]);
      ir::Value& out0 = hl_wd.CurrentIrValue();
      out0.SetNode(
          node_unpack,
          hl_wd.GetDevice(),
          hl_wd.GetSizes(),
          hl_wd.dtype_optional(),
          out_index++);
    }

    auto hl_exp_avg = GetHbLazyTensor(exp_avg[i]);
    ir::Value& out1 = hl_exp_avg.CurrentIrValue();
    out1.SetNode(
        node_unpack,
        hl_exp_avg.GetDevice(),
        hl_exp_avg.GetSizes(),
        hl_exp_avg.dtype_optional(),
        out_index++);

    auto hl_exp_avg_1 = GetHbLazyTensor(exp_avg[i]);
    ir::Value& out2 = hl_exp_avg_1.CurrentIrValue();
    out2.SetNode(
        node_unpack,
        hl_exp_avg_1.GetDevice(),
        hl_exp_avg_1.GetSizes(),
        hl_exp_avg_1.dtype_optional(),
        out_index++);

    auto hl_exp_avg_sq = GetHbLazyTensor(exp_avg_sq[i]);
    ir::Value& out3 = hl_exp_avg_sq.CurrentIrValue();
    out3.SetNode(
        node_unpack,
        hl_exp_avg_sq.GetDevice(),
        hl_exp_avg_sq.GetSizes(),
        hl_exp_avg_sq.dtype_optional(),
        out_index++);

    auto hl_exp_avg_sq_1 = GetHbLazyTensor(exp_avg_sq[i]);
    ir::Value& out4 = hl_exp_avg_sq_1.CurrentIrValue();
    out4.SetNode(
        node_unpack,
        hl_exp_avg_sq_1.GetDevice(),
        hl_exp_avg_sq_1.GetSizes(),
        hl_exp_avg_sq_1.dtype_optional(),
        out_index++);

    HbLazyTensorViews::CustomKernelAddNodeInplace(
        weights[i], node_unpack, out_index);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }
}

Tensor optimizer_lamb_fused_norm_hpu_lazy(
    const std::vector<at::Tensor>& grad,
    float max_grad_norm) {
  PT_LAZY_TRACE;
  auto clip_norm = get_tensor_for_scalar(1.0);
  ir::NodePtr node =
      std::make_shared<ir::LambFusedNorm>(grad, max_grad_norm, clip_norm);
  std::vector<int64_t> sizes{1};

  LazyOp<at::Tensor, ir::LambFusedNorm> k(
      node, {grad[0], max_grad_norm, clip_norm}, {sizes});

  return k.call();
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
  static_cast<void>(lr);

  /*
  Assuming mark step is present before optimizer is invoked
  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_grad = GetHbLazyTensor(gradients[i]);
    updateDstDependencies(hl_grad, gradients[i], true);

    auto hl_wts = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hl_wts, weights[i], true);

    auto hlexpavg = GetHbLazyTensor(exp_avg[i]);
    updateDstDependencies(hlexpavg, exp_avg[i], true);

    auto hlexpavgsq = GetHbLazyTensor(exp_avg_sq[i]);
    updateDstDependencies(hlexpavgsq, exp_avg_sq[i], true);
  }
  */

  auto hl_clip_global = GetHbLazyTensor(clip_global_grad_norm);
  updateDstDependencies(hl_clip_global, clip_global_grad_norm, true);

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
    ir::Value& out1 = hl_adam_step.CurrentIrValue();
    out1.SetNode(
        node,
        hl_adam_step.GetDevice(),
        hl_adam_step.GetSizes(),
        hl_adam_step.dtype_optional(),
        out_index++);

    context->m_retained_tensor_list.emplace_back(adam_step);
    adam_step_vec.push_back(adam_step);

    auto adam_norm = empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_adam_norm = GetHbLazyTensor(adam_norm);
    ir::Value& out2 = hl_adam_norm.CurrentIrValue();
    out2.SetNode(
        node,
        hl_adam_norm.GetDevice(),
        hl_adam_norm.GetSizes(),
        hl_adam_norm.dtype_optional(),
        out_index++);

    context->m_retained_tensor_list.emplace_back(adam_norm);
    adam_norm_vec.push_back(adam_norm);

    auto weight_norm = empty_hpu_lazy(
        {1}, weights[i].options(), weights[i].suggest_memory_format(), false);
    auto hl_weight_norm = GetHbLazyTensor(weight_norm);
    ir::Value& out3 = hl_weight_norm.CurrentIrValue();
    out3.SetNode(
        node,
        hl_weight_norm.GetDevice(),
        hl_weight_norm.GetSizes(),
        hl_weight_norm.dtype_optional(),
        out_index++);

    context->m_retained_tensor_list.emplace_back(weight_norm);
    weight_norm_vec.push_back(weight_norm);

    // add the tensors that are updated inplace
    auto exp_avg_temp = empty_hpu_lazy(
        exp_avg[i].sizes(),
        exp_avg[i].options(),
        exp_avg[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_temp = GetHbLazyTensor(exp_avg_temp);
    ir::Value& out4 = hl_exp_avg_temp.CurrentIrValue();
    out4.SetNode(
        node,
        hl_exp_avg_temp.GetDevice(),
        hl_exp_avg_temp.GetSizes(),
        hl_exp_avg_temp.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_temp);

    auto hl_exp_avg = GetHbLazyTensor(exp_avg[i]);
    ir::Value& out5 = hl_exp_avg.CurrentIrValue();
    out5.SetNode(
        node,
        hl_exp_avg.GetDevice(),
        hl_exp_avg.GetSizes(),
        hl_exp_avg.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg[i]);

    auto exp_avg_sq_temp = empty_hpu_lazy(
        exp_avg_sq[i].sizes(),
        exp_avg_sq[i].options(),
        exp_avg_sq[i].suggest_memory_format(),
        false);
    auto hl_exp_avg_sq_temp = GetHbLazyTensor(exp_avg_sq_temp);
    ir::Value& out6 = hl_exp_avg_sq_temp.CurrentIrValue();
    out6.SetNode(
        node,
        hl_exp_avg_sq_temp.GetDevice(),
        hl_exp_avg_sq_temp.GetSizes(),
        hl_exp_avg_sq_temp.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq_temp);

    auto hl_exp_avg_sq = GetHbLazyTensor(exp_avg_sq[i]);
    ir::Value& out7 = hl_exp_avg_sq.CurrentIrValue();
    out7.SetNode(
        node,
        hl_exp_avg_sq.GetDevice(),
        hl_exp_avg_sq.GetSizes(),
        hl_exp_avg_sq.dtype_optional(),
        out_index++);
    context->m_retained_tensor_list.emplace_back(exp_avg_sq[i]);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }

  return std::tie(weight_norm_vec, adam_norm_vec, adam_step_vec);
}

void optimizer_lamb_phase2_hpu_lazy(
    std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& adam_norm,
    const std::vector<at::Tensor>& weight_norm,
    const std::vector<at::Tensor>& adam_step,
    const std::vector<at::Tensor>& trust_ratio,
    const float step,
    const float weight_decay,
    const int use_lamb) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hl_weights = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hl_weights, weights[i], true);
  }

  auto nstep_t = at::tensor(-step).to(c10::kHPU, true);

  // Build Params for the graph
  ir::NodePtr node = std::make_shared<ir::OptimizerFusedLambPhase2>(
      weights,
      adam_norm,
      weight_norm,
      adam_step,
      trust_ratio,
      nstep_t,
      weight_decay,
      use_lamb);

  int64_t out_index = 0;
  for (size_t i = 0; i < weights.size(); i++) {
    HbLazyTensorViews::CustomKernelAddNodeInplace(weights[i], node, out_index);
  }
}

Tensor& optimizer_adagrad_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& variances,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float lrd,
    const float epsilon) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);

    auto hlvariance = GetHbLazyTensor(variances[i]);
    updateDstDependencies(hlvariance, variances[i], true);
  }

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedAdagrad>(
      gradients, weights, variances, epoch_num, lr, wd, lrd, epsilon);

  int64_t out_index = 0;
  HABANA_ASSERT(weights.size() == variances.size());
  for (size_t i = 0; i < weights.size(); i++) {
    HbLazyTensorViews::CustomKernelAddNodeInplace(weights[i], node, out_index);

    auto hlvariance = GetHbLazyTensor(variances[i]);
    ir::Value& out2 = hlvariance.CurrentIrValue();
    out2.SetNode(
        node,
        hlvariance.GetDevice(),
        hlvariance.GetSizes(),
        hlvariance.dtype_optional(),
        out_index++);
  }

  flush_op(lr);
  return lr;
}

Tensor& optimizer_sgd_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);
  }

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedSGD>(
      gradients, weights, lr, wd, mom, damp, nesterov);

  int64_t out_index = 0;

  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    ir::Value& out1 = hlweight.CurrentIrValue();
    out1.SetNode(
        node,
        hlweight.GetDevice(),
        hlweight.GetSizes(),
        hlweight.dtype_optional(),
        out_index++);
  }

  flush_op(lr);
  return lr;
}

Tensor& optimizer_sgd_momentum_hpu_lazy(
    const TensorList& gradients,
    TensorList& weights,
    TensorList& momentum,
    const at::Tensor& epoch_num,
    at::Tensor& lr,
    const float wd,
    const float mom,
    const float damp,
    const bool nesterov) {
  PT_LAZY_TRACE;
  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    updateDstDependencies(hlweight, weights[i], true);

    auto hlmomentum = GetHbLazyTensor(momentum[i]);
    updateDstDependencies(hlmomentum, momentum[i], true);
  }

  ir::NodePtr node = std::make_shared<ir::OptimizerFusedSGDMomentum>(
      gradients, weights, momentum, epoch_num, lr, wd, mom, damp, nesterov);

  int64_t out_index = 0;
  HABANA_ASSERT(weights.size() == momentum.size());

  auto hlweight = GetHbLazyTensor(weights[0]);
  ir::Value& out = hlweight.CurrentIrValue();
  node->set_as_output_tensor_list();
  out.SetNode(
      node,
      hlweight.GetDevice(),
      hlweight.GetSizes(),
      hlweight.dtype_optional());

  ir::NodePtr node_unpack = std::make_shared<ir::ListUnpack>(out);
  PT_BRIDGE_DEBUG("FE weights & mumentum tensors size: ", weights.size());
  for (size_t i = 0; i < weights.size(); i++) {
    auto hlweight = GetHbLazyTensor(weights[i]);
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
      if (hlweight.GetHbLazyTensorData().has_value()) {
        auto internal_tensor = hlweight.GetHbLazyTensorData().value();
        auto hb_lazy_impl_internal = GetHbInternalTensorImpl(internal_tensor);
        if (hb_lazy_impl_internal) {
          PT_BRIDGE_DEBUG(
              "Optimizer FE weight tensor with address: ",
              hb_lazy_impl_internal,
              " permutation: ",
              VecToString(hb_lazy_impl_internal->GetMemoryPermutation()));
        } else {
          PT_BRIDGE_DEBUG(
              "Optimizer FE weight tensor has no BE tensor, this could indicate a problem");
        }
      }
    }
    HbLazyTensorViews::CustomKernelAddNodeInplace(
        weights[i], node_unpack, out_index);

    auto hlmomentum = GetHbLazyTensor(momentum[i]);
    if (GET_ENV_FLAG_NEW(PT_HPU_ENABLE_SYNAPSE_LAYOUT_HANDLING)) {
      if (hlmomentum.GetHbLazyTensorData().has_value()) {
        auto internal_tensor = hlmomentum.GetHbLazyTensorData().value();
        auto hb_lazy_impl_internal = GetHbInternalTensorImpl(internal_tensor);
        if (hb_lazy_impl_internal) {
          PT_BRIDGE_DEBUG(
              "Optimizer FE momentum tensor with address: ",
              hb_lazy_impl_internal,
              " permutation: ",
              VecToString(hb_lazy_impl_internal->GetMemoryPermutation()));
        } else {
          PT_BRIDGE_DEBUG(
              "Optimizer FE momentum tensor has no BE tensor, this could indicate a problem");
        }
      }
    }
    ir::Value& out2 = hlmomentum.CurrentIrValue();
    out2.SetNode(
        node_unpack,
        hlmomentum.GetDevice(),
        hlmomentum.GetSizes(),
        hlmomentum.dtype_optional(),
        out_index++);
  }

  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 2) {
    HbLazyTensor::StepMarker({});
  }
  return lr;
}

} // namespace habana_lazy
