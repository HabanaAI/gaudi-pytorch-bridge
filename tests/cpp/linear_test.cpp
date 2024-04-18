/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
class LazyLinearTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyLinearTest, LinearBwdTest) {
  int out_features = 4;
  int in_features = 2;
  int m = 2;
  int n = 2;

  auto in = torch::randn({in_features}, torch::requires_grad());
  auto hin = in.to(torch::kHPU);
  auto wt =
      torch::randn({out_features, in_features}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features}, torch::requires_grad());

  auto exp = torch::linear(in, wt);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << exp.cpu() << std::endl;
  auto exp_hpu = torch::linear(hin, hwt);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);
  // exp_hpu.backward(hgrad_out);

  auto grad_in = in.grad();
  auto grad_wt = wt.grad();

  at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 0};
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
      torch::linear_backward(hin, hgrad_out, hwt, mask);

  // HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  std::cout << "grad_wt" << grad_wt << std::endl;
  std::cout << "hgrad_wt_cpu" << hgrad_wt_cpu << std::endl;
  std::cout << "grad_in" << grad_in << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in << std::endl;
  EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
}