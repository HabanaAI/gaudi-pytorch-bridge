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
    double momentum) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<torch::Tensor> params_momentum_buf_list;
    std::vector<torch::Tensor> dp_list;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    auto params_in = torch::randn({M, N});
    dump_tensor<float>(
        "params_in[" + std::to_string(i) + "]", params_in, verbose);
    PushBackHpuAndCpuTensors(
        params_in, hpu, cpu, &Data::params_momentum_buf_list);

    auto momentum_in = torch::randn({M, N});
    dump_tensor<float>(
        "momentum_in[" + std::to_string(i) + "]", momentum_in, verbose);
    PushBackHpuAndCpuTensors(
        momentum_in, hpu, cpu, &Data::params_momentum_buf_list);

    auto dp_in = torch::randn({M, N});
    dump_tensor<float>("dp_in[" + std::to_string(i) + "]", dp_in, verbose);
    PushBackHpuAndCpuTensors(dp_in, hpu, cpu, &Data::dp_list);
  }

  torch::TensorList params_momentum_buf_list(hpu.params_momentum_buf_list);
  torch::TensorList dp_list(hpu.dp_list);

  optimizer_resource_apply_momentum_hpu_wrap(
      params_momentum_buf_list, dp_list, momentum);

  // CPU calculations
  for (auto i = 0; i < num_params; i++) {
    const auto i2 = 2 * i;
    const auto i2p1 = i2 + 1;

    cpu.params_momentum_buf_list[i2p1].mul_(momentum).sub_(cpu.dp_list[i]);
    cpu.params_momentum_buf_list[i2].add_(cpu.params_momentum_buf_list[i2p1]);
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
    bool grads_zero) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<torch::Tensor> params;
    std::vector<torch::Tensor> grads;
  } cpu, hpu;

  std::vector<long> shape =
      (N > 1) ? std::vector<long>{M, N} : std::vector<long>{M};

  for (auto i = 0; i < num_params; ++i) {
    auto params_in = params_zero ? torch::zeros(shape) : torch::randn(shape);
    dump_tensor<float>(
        "params_in[" + std::to_string(i) + "]", params_in, verbose);
    PushBackHpuAndCpuTensors(params_in, hpu, cpu, &Data::params);

    auto grads_in = grads_zero ? torch::zeros(shape) : torch::randn(shape);
    dump_tensor<float>(
        "grads_in[" + std::to_string(i) + "]", grads_in, verbose);
    PushBackHpuAndCpuTensors(grads_in, hpu, cpu, &Data::grads);
  }

  torch::TensorList params(hpu.params);
  torch::TensorList grads(hpu.grads);

  optimizer_lars_hpu_wrap(
      params, grads, skip_masks, eeta, weight_decay, eps, lr);

  // CPU calculations
  for (auto i = 0; i < num_params; i++) {
    if (!skip_masks[i]) {
      cpu.grads[i].mul_(lr);
    } else {
      auto params_norm = cpu.params[i].square().sum().sqrt();
      auto grads_norm = cpu.grads[i].square().sum().sqrt();
      auto params_norm_positive = params_norm.greater(0.0);
      auto grads_norm_positive = grads_norm.greater(0.0);
      auto nominator = params_norm.mul(eeta);
      auto denominator = params_norm.mul(weight_decay).add(eps).add(grads_norm);
      auto division = nominator.div(denominator);
      auto selection = torch::where(
          grads_norm_positive,
          torch::where(params_norm_positive, division, 1.0),
          1.0);
      cpu.grads[i] = cpu.params[i]
                         .mul(weight_decay)
                         .add(cpu.grads[i])
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
    std::vector<torch::Tensor> weights_vec;
    std::vector<torch::Tensor> adam_norms_vec;
    std::vector<torch::Tensor> weight_norms_vec;
    std::vector<torch::Tensor> adam_steps_vec;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    auto weight = with_view ? torch::randn({M * N}) : torch::randn({M, N});
    dump_tensor<float>("weight_in[" + std::to_string(i) + "]", weight, verbose);
    PushBackHpuAndCpuTensors(weight, hpu, cpu, &Data::weights_vec);
    if (with_view) {
      cpu.weights_vec[i] = cpu.weights_vec[i].view({M, N});
      hpu.weights_vec[i] = hpu.weights_vec[i].view({M, N});
    }

    auto adam_norm = torch::rand({1});
    dump_tensor<float>(
        "adam_norm_in[" + std::to_string(i) + "]", adam_norm, verbose);
    PushBackHpuAndCpuTensors(adam_norm, hpu, cpu, &Data::adam_norms_vec);

    auto weight_norm = torch::rand({1});
    dump_tensor<float>(
        "weight_norm_in[" + std::to_string(i) + "]", weight_norm, verbose);
    PushBackHpuAndCpuTensors(weight_norm, hpu, cpu, &Data::weight_norms_vec);

    auto adam_step = torch::randn({M, N});
    dump_tensor<float>(
        "adam_step_in[" + std::to_string(i) + "]", adam_step, verbose);
    PushBackHpuAndCpuTensors(adam_step, hpu, cpu, &Data::adam_steps_vec);
  }

  torch::TensorList weights(hpu.weights_vec);
  torch::TensorList adam_norms(hpu.adam_norms_vec);
  torch::TensorList weight_norms(hpu.weight_norms_vec);
  torch::TensorList adam_steps(hpu.adam_steps_vec);

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
        (cpu.adam_norms_vec[i][0].item<float>() > 0) &&
        (cpu.weight_norms_vec[i][0].item<float>() > 0)) {
      trust_ratio = cpu.weight_norms_vec[i] / cpu.adam_norms_vec[i];
    }
    cpu.adam_steps_vec[i] *= -step * trust_ratio;
    cpu.weights_vec[i] =
        torch::add(cpu.weights_vec[i], cpu.adam_steps_vec[i], 1.0);
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