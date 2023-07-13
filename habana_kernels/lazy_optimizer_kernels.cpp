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
