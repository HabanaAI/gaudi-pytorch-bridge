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
#include "hpu_ops/hpu_op_helper.h"

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

  PT_IRGRAPH_DEBUG(
      "step marker due to optimizer_sparse_sgd_with_valid_count_hpu_lazy");
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

void optimizer_ema_hpu_lazy(
    const at::TensorList model_inputs,
    at::TensorList updated_ema,
    const at::Tensor& decay) {
  PT_LAZY_TRACE;
  habana_lazy::NoAccThread no_acc_thread;

  ir::NodePtr node =
      std::make_shared<ir::OptimizerFusedEMA>(model_inputs, updated_ema, decay);

  int64_t out_index = 0;

  auto hl_ema = GetHbLazyTensor(updated_ema[0]);
  node->set_as_output_tensor_list();
  ir::Value& out = hl_ema.IrSetNode(node);
  ir::NodePtr node_unpack = std::make_shared<ir::ListUnpack>(out);

  for (size_t i = 0; i < updated_ema.size(); i++) {
    HbLazyTensorViews::CustomKernelAddNodeInplace(
        updated_ema[i], node_unpack, out_index);
  }

  flush_op();
}

static void CastMomentToFp8WithScale(
    const at::Tensor& scaled_moment,
    at::Tensor& scale,
    at::Tensor& fp8_moment,
    const float fp8_max) {
  at::Tensor amax = at::max(at::abs(scaled_moment));
  at::Tensor exp = at::floor(at::log2(fp8_max / amax));
  at::Tensor sf = at::pow(2.0, exp);
  at::where_out(scale, amax > 0.0, sf, scale);

  auto cast_results = cast_to_fp8_v2_lazy(
      scaled_moment, scale, true, false, fp8_moment.scalar_type(), {1});
  copy_hpu_lazy_(fp8_moment, std::get<0>(cast_results), true);
}

void optimizer_adamw_hpu_lazy(
    const at::TensorList gradients,
    at::TensorList weights,
    at::TensorList exp_avg,
    at::TensorList exp_avg_sq,
    const at::Tensor& neg_step_t,
    const double beta1,
    const double beta2,
    const double epsilon,
    const double modified_wd,
    c10::optional<at::TensorList> exp_avg_scales,
    c10::optional<at::TensorList> exp_avg_sq_scales) {
  PT_LAZY_TRACE;
  std::vector<at::Tensor> gradients_v;
  std::vector<at::Tensor> weights_v;
  std::vector<at::Tensor> exp_avg_v;
  std::vector<at::Tensor> exp_avg_sq_v;
  std::vector<at::Tensor> exp_avg_scales_v;
  std::vector<at::Tensor> exp_avg_sq_scales_v;

  std::copy(
      gradients.begin(), gradients.end(), std::back_inserter(gradients_v));
  std::copy(weights.begin(), weights.end(), std::back_inserter(weights_v));
  std::copy(exp_avg.begin(), exp_avg.end(), std::back_inserter(exp_avg_v));
  std::copy(
      exp_avg_sq.begin(), exp_avg_sq.end(), std::back_inserter(exp_avg_sq_v));

  if (exp_avg_scales.has_value()) {
    std::copy(
        exp_avg_scales.value().begin(),
        exp_avg_scales.value().end(),
        std::back_inserter(exp_avg_scales_v));
    handle_collective(exp_avg_scales_v);
  }
  if (exp_avg_sq_scales.has_value()) {
    std::copy(
        exp_avg_sq_scales.value().begin(),
        exp_avg_sq_scales.value().end(),
        std::back_inserter(exp_avg_sq_scales_v));
    handle_collective(exp_avg_sq_scales_v);
  }

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
               neg_step_t,
               beta1,
               beta2,
               epsilon,
               modified_wd_t,
               is_wd_modified,
               exp_avg_scales_v = std::move(exp_avg_scales_v),
               exp_avg_sq_scales_v = std::move(exp_avg_sq_scales_v)]() mutable {
    const bool is_fp8 = exp_avg_scales_v.size() != 0;

    std::vector<at::Tensor> exp_avg_scaled;
    std::vector<at::Tensor> exp_avg_sq_scaled;
    for (size_t i = 0; i < exp_avg_scales_v.size(); i++) {
      exp_avg_scaled.push_back(cast_from_fp8_lazy(
          exp_avg_v[i],
          exp_avg_scales_v[i],
          gradients_v[i].scalar_type(),
          c10::nullopt));
      exp_avg_sq_scaled.push_back(cast_from_fp8_lazy(
          exp_avg_sq_v[i],
          exp_avg_sq_scales_v[i],
          gradients_v[i].scalar_type(),
          c10::nullopt));
    }

    TensorList gradients = gradients_v;
    TensorList weights = weights_v;
    TensorList exp_avg = is_fp8 ? exp_avg_scaled : exp_avg_v;
    TensorList exp_avg_sq = is_fp8 ? exp_avg_sq_scaled : exp_avg_sq_v;

    auto hl_neg_step_t = GetHbLazyTensor(neg_step_t);
    auto hl_modified_wd_t = GetHbLazyTensor(modified_wd_t);

    ir::NodePtr node = std::make_shared<ir::OptimizerFusedAdamw>(
        gradients,
        weights,
        exp_avg,
        exp_avg_sq,
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

    if (is_fp8) {
      const float FP8_E4M3_MAX = 240.0;
      const float FP8_E5M2_MAX = 57344.0;
      for (size_t i = 0; i < exp_avg_v.size(); i++) {
        CastMomentToFp8WithScale(
            exp_avg_scaled[i],
            exp_avg_scales_v[i],
            exp_avg_v[i],
            exp_avg_v[i].scalar_type() == ScalarType::Float8_e4m3fn
                ? FP8_E4M3_MAX
                : FP8_E5M2_MAX);

        CastMomentToFp8WithScale(
            exp_avg_sq_scaled[i],
            exp_avg_sq_scales_v[i],
            exp_avg_sq_v[i],
            exp_avg_sq_v[i].scalar_type() == ScalarType::Float8_e4m3fn
                ? FP8_E4M3_MAX
                : FP8_E5M2_MAX);
      }
    }
  };
  auto vector_of_inputs = std::vector<c10::IValue>{
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      modified_wd_t,
      is_wd_modified,
      exp_avg_scales,
      exp_avg_sq_scales};
  RUNNING_HASH_COMBINE_OPERATOR(hpu::habanaOptimizerAdamW, vector_of_inputs);
  RUN_MANUAL_OP_NO_RETURN_WITH_ACC_THREAD(optimizer_adamw, func)
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
      "hpu::optimizer_sgd", {gradients, weights, lr, wd, mom, damp, nesterov});

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
      "hpu::optimizer_sgd_momentum",
      {gradients, weights, momentum, epoch_num, lr, mom, wd, damp, nesterov});
  loo.call(weights, momentum, OPTIMIZER::SGD_MOMENTUM);
}

void optimizer_lars_hpu_lazy(
    const at::TensorList params,
    at::TensorList grads,
    const std::vector<int64_t> skipMasks,
    const float eeta,
    const float weight_decay,
    const float eps,
    const float lr) {
  auto lr_t = get_tensor_for_scalar(lr, params[0].options());
  std::vector<at::Tensor> params_copy;
  std::copy(params.begin(), params.end(), std::back_inserter(params_copy));
  std::vector<at::Tensor> grads_copy;
  std::copy(grads.begin(), grads.end(), std::back_inserter(grads_copy));

  handle_collective(params);
  handle_collective(grads);

  auto func = [grads_copy = std::move(grads_copy),
               params_copy = std::move(params_copy),
               skipMasks,
               eeta,
               weight_decay,
               eps,
               lr_t]() {
    auto params = torch::TensorList(params_copy);
    auto grads = torch::TensorList(grads_copy);
    LazyOptimizationOp<void> lo(
        "hpu::habanaOptimizerLars",
        {grads, params, lr_t, skipMasks, eeta, weight_decay, eps});
    lo.call(grads, LARS);
  };
  RUN_MANUAL_OP_NO_RETURN_WITH_ACC_THREAD(optimizer_lars, func);
}

} // namespace habana_lazy
