/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include <stdexcept>

#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

#include "backend/synapse_helpers/env_flags.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;

class LazyControlEdgeTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyControlEdgeTest, AsStridedWithinOps) {
  torch::Tensor c0 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c1 = torch::as_strided(c0, {20, 5}, {5, 1});

  torch::Tensor c2 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c3 = c2.abs();
  torch::Tensor c4 = torch::as_strided(c3, {20, 5}, {5, 1});
  c4.copy_(c1);
  torch::Tensor c5 = c3.relu();

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = torch::as_strided(h0, {20, 5}, {5, 1});
  torch::Tensor h2 = c2.to(torch::kHPU);
  torch::Tensor h3 = h2.abs();
  torch::Tensor h4 = torch::as_strided(h3, {20, 5}, {5, 1});
  h4.copy_(h1);
  torch::Tensor h5 = h3.relu();

  torch::Tensor h5_c = h5.to(torch::kCPU);

  EXPECT_TRUE(allclose(c5, h5_c));
}

TEST_F(LazyControlEdgeTest, ControlEdgeCycle) {
  auto c0 = torch::tensor({2.0, 3.0});
  auto c1 = c0.squeeze(-1);
  auto c2 = c1.mul(0.1);
  auto c3 = torch::mul(c0, c2);

  auto h0 = c0.to(torch::kHPU);
  auto h1 = h0.squeeze(-1);
  auto h2 = h1.mul(0.1);
  auto h3 = torch::mul(h0, h2);

  constexpr float rtol = 1e-3;
  constexpr float atol = 1e-3;
  auto h3_c = h3.to(torch::kCPU);
  EXPECT_TRUE(allclose(c3, h3_c, rtol, atol));
}

TEST_F(LazyControlEdgeTest, stridedinsertreuse) {
  torch::Tensor A = torch::randn({4});
  auto b = torch::relu(A);
  auto v1 = A.view(-1);
  auto grad1 = torch::randn({4});

  auto hA = A.to(torch::kHPU);
  auto hB = torch::relu(hA);
  auto hv1 = hA.view(-1);
  auto hgrad1 = grad1.to(torch::kHPU);

  v1.mul_(grad1);

  hv1.mul_(hgrad1);

  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
}
