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

    auto momentum_in = torch::randn({M, N});
    dump_tensor<float>(
        "momentum_in[" + std::to_string(i) + "]", momentum_in, verbose);

    auto dp_in = torch::randn({M, N});
    dump_tensor<float>("dp_in[" + std::to_string(i) + "]", dp_in, verbose);

    cpu.params_momentum_buf_list.push_back(params_in);
    auto paramsH = params_in.to(torch::kHPU);
    hpu.params_momentum_buf_list.push_back(paramsH);

    cpu.params_momentum_buf_list.push_back(momentum_in);
    auto momentumH = momentum_in.to(torch::kHPU);
    hpu.params_momentum_buf_list.push_back(momentumH);

    cpu.dp_list.push_back(dp_in);
    auto dpH = dp_in.to(torch::kHPU);
    hpu.dp_list.push_back(dpH);
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