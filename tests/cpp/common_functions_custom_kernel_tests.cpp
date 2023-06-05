/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "common_functions_custom_kernel_tests.h"
#include <gtest/gtest.h>
#include "common_functions_helpers.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"

void runResourceApplyMomentumOptTest(
    int num_params,
    int M,
    int N,
    double momentum,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> params_momentum_buf_list;
    std::vector<TensorAndView> dp_list;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto params_in = torch::randn({M, N});
    dump_tensor<float>(
        "params_in[" + std::to_string(i) + "]", params_in, verbose);
    PushBackHpuAndCpuTensors(
        params_in, hpu, cpu, &Data::params_momentum_buf_list, use_views);

    auto momentum_in = torch::randn({M, N});
    dump_tensor<float>(
        "momentum_in[" + std::to_string(i) + "]", momentum_in, verbose);
    PushBackHpuAndCpuTensors(
        momentum_in, hpu, cpu, &Data::params_momentum_buf_list, use_views);

    auto dp_in = torch::randn({M, N});
    dump_tensor<float>("dp_in[" + std::to_string(i) + "]", dp_in, verbose);
    PushBackHpuAndCpuTensors(dp_in, hpu, cpu, &Data::dp_list, use_views);
  }

  auto params_momentum_buf_list =
      TensorAndViewVecToViewVec(hpu.params_momentum_buf_list);
  auto dp_list = TensorAndViewVecToViewVec(hpu.dp_list);

  optimizer_resource_apply_momentum_hpu_wrap(
      params_momentum_buf_list, dp_list, momentum);

  // CPU calculations
  for (auto i = 0; i < num_params; i++) {
    const auto i2 = 2 * i;
    const auto i2p1 = i2 + 1;

    cpu.params_momentum_buf_list[i2p1].t.mul_(momentum).sub_(cpu.dp_list[i].t);
    cpu.params_momentum_buf_list[i2].t.add_(
        cpu.params_momentum_buf_list[i2p1].t);
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "params_list",
        std::make_pair(&Data::params_momentum_buf_list, 2 * i),
        "momentum_list",
        std::make_pair(&Data::params_momentum_buf_list, 2 * i + 1),
        "dp_list",
        &Data::dp_list);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}

void runLarsOptTest(
    int num_params,
    int M,
    int N,
    const std::vector<int64_t>& skip_masks,
    double eeta,
    double weight_decay,
    double eps,
    double lr,
    bool params_zero,
    bool grads_zero,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> params;
    std::vector<TensorAndView> grads;
  } cpu, hpu;

  std::vector<long> shape =
      (N > 1) ? std::vector<long>{M, N} : std::vector<long>{M};

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto params_in = params_zero ? torch::zeros(shape) : torch::randn(shape);
    dump_tensor<float>(
        "params_in[" + std::to_string(i) + "]", params_in, verbose);
    PushBackHpuAndCpuTensors(params_in, hpu, cpu, &Data::params, use_views);

    auto grads_in = grads_zero ? torch::zeros(shape) : torch::randn(shape);
    dump_tensor<float>(
        "grads_in[" + std::to_string(i) + "]", grads_in, verbose);
    PushBackHpuAndCpuTensors(grads_in, hpu, cpu, &Data::grads, use_views);
  }

  auto params = TensorAndViewVecToViewVec(hpu.params);
  auto grads = TensorAndViewVecToViewVec(hpu.grads);

  auto lr_t = torch::full({1}, lr).to("hpu");

  optimizer_lars_hpu_wrap(
      params, grads, skip_masks, eeta, weight_decay, eps, lr_t);

  // CPU calculations
  for (auto i = 0; i < num_params; i++) {
    if (!skip_masks[i]) {
      cpu.grads[i].t.mul_(lr);
    } else {
      auto params_norm = cpu.params[i].t.square().sum().sqrt();
      auto grads_norm = cpu.grads[i].t.square().sum().sqrt();
      auto params_norm_positive = params_norm.greater(0.0);
      auto grads_norm_positive = grads_norm.greater(0.0);
      auto nominator = params_norm.mul(eeta);
      auto denominator = params_norm.mul(weight_decay).add(eps).add(grads_norm);
      auto division = nominator.div(denominator);
      auto selection = torch::where(
          grads_norm_positive,
          torch::where(params_norm_positive, division, 1.0),
          1.0);
      cpu.grads[i].t = cpu.params[i]
                           .t.mul(weight_decay)
                           .add(cpu.grads[i].t)
                           .mul(selection.mul(lr));
    }
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "params",
        &Data::params,
        "grads",
        &Data::grads);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}

void runLambPhase2OptimizerTest(
    int num_params,
    int M,
    int N,
    const double weight_decay,
    const bool use_lamb,
    const bool with_view) {
  torch::manual_seed(0);
  bool verbose = false;
  float step = 0.1;

  struct Data {
    std::vector<TensorAndView> weights_vec;
    std::vector<TensorAndView> adam_norms_vec;
    std::vector<TensorAndView> weight_norms_vec;
    std::vector<TensorAndView> adam_steps_vec;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    auto weight = torch::randn({M, N});
    dump_tensor<float>("weight_in[" + std::to_string(i) + "]", weight, verbose);
    PushBackHpuAndCpuTensors(weight, hpu, cpu, &Data::weights_vec, with_view);

    auto adam_norm = torch::rand({1});
    dump_tensor<float>(
        "adam_norm_in[" + std::to_string(i) + "]", adam_norm, verbose);
    PushBackHpuAndCpuTensors(adam_norm, hpu, cpu, &Data::adam_norms_vec, false);

    auto weight_norm = torch::rand({1});
    dump_tensor<float>(
        "weight_norm_in[" + std::to_string(i) + "]", weight_norm, verbose);
    PushBackHpuAndCpuTensors(
        weight_norm, hpu, cpu, &Data::weight_norms_vec, false);

    auto adam_step = torch::randn({M, N});
    dump_tensor<float>(
        "adam_step_in[" + std::to_string(i) + "]", adam_step, verbose);
    PushBackHpuAndCpuTensors(
        adam_step, hpu, cpu, &Data::adam_steps_vec, with_view);
  }

  auto weights = TensorAndViewVecToViewVec(hpu.weights_vec);
  auto adam_norms = TensorAndViewVecToViewVec(hpu.adam_norms_vec);
  auto weight_norms = TensorAndViewVecToViewVec(hpu.weight_norms_vec);
  auto adam_steps = TensorAndViewVecToViewVec(hpu.adam_steps_vec);

  habana_lazy::optimizer_lamb_fused_phase2(
      weights,
      adam_norms,
      weight_norms,
      adam_steps,
      step,
      weight_decay,
      use_lamb);

  // CPU calculations
  for (int i = 0; i < num_params; i++) {
    torch::Tensor trust_ratio = torch::ones(1);
    if ((weight_decay != 0 || use_lamb) &&
        (cpu.adam_norms_vec[i].t[0].item<float>() > 0) &&
        (cpu.weight_norms_vec[i].t[0].item<float>() > 0)) {
      trust_ratio = cpu.weight_norms_vec[i].t / cpu.adam_norms_vec[i].t;
    }
    cpu.adam_steps_vec[i].t *= -step * trust_ratio;
    cpu.weights_vec[i].t =
        torch::add(cpu.weights_vec[i].t, cpu.adam_steps_vec[i].t, 1.0);
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i, hpu, cpu, verbose, 1e-06, 1e-06, "weights_vec", &Data::weights_vec);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}

void runEmaOptTest(
    int num_params,
    int M,
    int N,
    double decay_val,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> model_inputs;
    std::vector<TensorAndView> updated_ema;
    std::vector<TensorAndView> decay;
  } cpu, hpu;

  auto decay_in = torch::full({1}, decay_val);
  dump_tensor<float>("decay_in", decay_in, verbose);
  PushBackHpuAndCpuTensors(decay_in, hpu, cpu, &Data::decay, false);

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto model_inputs_in = torch::randn({M, N});
    dump_tensor<float>(
        "model_inputs_in[" + std::to_string(i) + "]", model_inputs_in, verbose);
    PushBackHpuAndCpuTensors(
        model_inputs_in, hpu, cpu, &Data::model_inputs, use_views);

    auto updated_ema_in = torch::randn({M, N});
    dump_tensor<float>(
        "updated_ema_in[" + std::to_string(i) + "]", updated_ema_in, verbose);
    PushBackHpuAndCpuTensors(
        updated_ema_in, hpu, cpu, &Data::updated_ema, use_views);
  }

  auto model_inputs = TensorAndViewVecToViewVec(hpu.model_inputs);
  auto updated_ema = TensorAndViewVecToViewVec(hpu.updated_ema);
  auto decay = TensorAndViewVecToViewVec(hpu.decay)[0];

  optimizer_ema_hpu_wrap(model_inputs, updated_ema, decay);

  // CPU calculations
  auto one_minus_decay = 1.0 - cpu.decay[0].t;
  for (auto i = 0; i < num_params; i++) {
    cpu.updated_ema[i].t.mul_(cpu.decay[0].t);
    cpu.updated_ema[i].t.add_(cpu.model_inputs[i].t.mul(one_minus_decay));
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "model_inputs",
        &Data::model_inputs,
        "updated_ema",
        &Data::updated_ema);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}