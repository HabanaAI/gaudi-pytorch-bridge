/******************************************************************************
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

#include <iostream>
#include "util.h"

class NativeGroupNormTests : public HpuOpTestUtil {};

TEST_F(NativeGroupNormTests, GroupNormFwdBwdExecute) {
  const int64_t N = 2;
  const int64_t C = 4;
  const int64_t H = 8;
  const int64_t W = 16;
  const int64_t G = 4;
  double eps = 0.0001;
  auto input_tensor =
      torch::arange(
          N * C * H * W, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({N, C, H, W}); // nchw

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);

  at::Tensor weight =
      torch::ones(C, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({C}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHPU);

  at::Tensor bias =
      torch::zeros(C, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({C}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHPU);

  auto results_cpu =
      torch::native_group_norm(input_tensor, weight, bias, N, C, H * W, G, eps);
  at::Tensor mean_cpu = std::get<1>(results_cpu);
  at::Tensor rstd_cpu = std::get<2>(results_cpu);

  at::Tensor tHabanaMean = std::get<1>(results_cpu).to(torch::kHPU);
  at::Tensor tHabanaRstd = std::get<2>(results_cpu).to(torch::kHPU);

  auto input_grad =
      torch::ones(
          N * C * H * W, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({N, C, H, W}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHPU);

  auto results_bwd = torch::native_group_norm_backward(
      tHabanaGrad,
      tHabanaX,
      tHabanaMean,
      tHabanaRstd,
      tWeight,
      N,
      C,
      H * W,
      G,
      {true, true, true});

  auto results_bwd_cpu = torch::native_group_norm_backward(
      input_grad,
      input_tensor,
      mean_cpu,
      rstd_cpu,
      weight,
      N,
      C,
      H * W,
      G,
      {true, true, true});
  at::Tensor result_bwd_lazy = std::get<0>(results_bwd);
  at::Tensor grad_weight_bwd_lazy = std::get<1>(results_bwd);
  at::Tensor grad_bias_bwd_lazy = std::get<2>(results_bwd);

  at::Tensor result_bwd_cpu = std::get<0>(results_bwd_cpu);
  at::Tensor grad_weight_bwd_cpu = std::get<1>(results_bwd_cpu);
  at::Tensor grad_bias_bwd_cpu = std::get<2>(results_bwd_cpu);
  Compare(result_bwd_cpu, result_bwd_lazy);
  Compare(grad_weight_bwd_cpu, grad_weight_bwd_lazy);
  Compare(grad_bias_bwd_cpu, grad_bias_bwd_lazy);
}