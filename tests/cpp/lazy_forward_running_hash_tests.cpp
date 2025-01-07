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
#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
using namespace habana_lazy;
using namespace at;

class LazyFwdRunningHashTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyFwdRunningHashTest, remove_post_order_test) {
  // Graph1
  torch::Tensor A = torch::randn({2, 2});
  torch::Tensor B = torch::randn({2, 2});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = torch::add(hA, hB);
  auto hD1 = torch::relu(hC);
  auto hD = torch::t(hD1);
  auto out = hD.to(kCPU);

  // cpu
  auto C = torch::add(A, B);
  auto D1 = torch::relu(C);
  auto outcpu = torch::t(D1);
  EXPECT_EQ(allclose(out, outcpu, 1e-03, 1e-03), true);

  // run twice and check if we are generating smae hash above
  for (size_t idx = 0; idx < 2; idx++) {
    torch::Tensor A = torch::randn({2, 2});
    torch::Tensor B = torch::randn({2, 2});
    auto hA = A.to(torch::kHPU);
    auto hB = B.to(torch::kHPU);
    auto hC = torch::add(hA, hB);
    auto hD1 = torch::relu(hC);
    auto hD = torch::t(hD1);
    auto out = hD.to(kCPU);
    // cpu
    auto C = torch::add(A, B);
    auto D1 = torch::relu(C);
    auto outcpu = torch::t(D1);
    EXPECT_EQ(allclose(out, outcpu, 1e-03, 1e-03), true);
  }

  // Graph2
  // create a new graph only change the order of nodes see if we get new hash
  if (1) {
    torch::Tensor A = torch::randn({2, 2});
    torch::Tensor B = torch::randn({2, 2});
    auto hA = A.to(torch::kHPU);
    auto hB = B.to(torch::kHPU);
    auto hC = torch::relu(hA);
    auto hD1 = torch::add(hC, hB);
    auto hD = torch::t(hD1);
    auto out = hD.to(kCPU);
    // cpu
    auto C = torch::relu(A);
    auto D1 = torch::add(C, B);
    auto outcpu = torch::t(D1);
    EXPECT_EQ(allclose(out, outcpu, 1e-03, 1e-03), true);

    // run twice and check if we are generating smae hash above
    for (size_t idx = 0; idx < 2; idx++) {
      torch::Tensor A = torch::randn({2, 2});
      torch::Tensor B = torch::randn({2, 2});
      auto hA = A.to(torch::kHPU);
      auto hB = B.to(torch::kHPU);
      auto hC = torch::relu(hA);
      auto hD1 = torch::add(hC, hB);
      auto hD = torch::t(hD1);
      auto out = hD.to(kCPU);
      // cpu
      auto C = torch::relu(A);
      auto D1 = torch::add(C, B);
      auto outcpu = torch::t(D1);
      EXPECT_EQ(allclose(out, outcpu, 1e-03, 1e-03), true);
    }
  }
  {
    {
      // Graph1
      torch::Tensor A = torch::randn({2, 2});
      torch::Tensor B = torch::randn({2, 2});
      auto hA = A.to(torch::kHPU);
      auto hB = B.to(torch::kHPU);
      auto hC = torch::add(hA, hB);
      auto hD1 = torch::relu(hC);
      auto hD = torch::t(hD1);
      auto out = hD.to(kCPU);

      // cpu
      auto C = torch::add(A, B);
      auto D1 = torch::relu(C);
      auto outcpu = torch::t(D1);
      EXPECT_EQ(allclose(out, outcpu, 1e-03, 1e-03), true);
    }

    {
      // Graph2
      torch::Tensor A = torch::randn({2, 2});
      torch::Tensor B = torch::randn({2, 2});
      auto hA = A.to(torch::kHPU);
      auto hB = B.to(torch::kHPU);
      auto hC = torch::relu(hA);
      auto hD1 = torch::add(hC, hB);
      auto hD = torch::t(hD1);
      auto out = hD.to(kCPU);

      // cpu
      auto C = torch::relu(A);
      auto D1 = torch::add(C, B);
      auto outcpu = torch::t(D1);
      EXPECT_EQ(allclose(out, outcpu, 1e-03, 1e-03), true);
    }
  }
}

TEST_F(LazyFwdRunningHashTest, remove_post_order_scope_test) {
  // graph 1
  torch::Tensor A = torch::randn({2, 2});
  torch::Tensor B = torch::randn({2, 2});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = torch::add(hA, hB);
  auto hD = torch::relu(hC);
  // cpu output
  auto C = torch::add(A, B);
  auto D = torch::relu(C);

  // graph 2 in scope
  {
    torch::Tensor A1 = torch::randn({2, 2});
    torch::Tensor B1 = torch::randn({2, 2});
    auto hA1 = A1.to(torch::kHPU);
    auto hB1 = B1.to(torch::kHPU);
    auto hC1 = torch::tanh(hA1);
    auto hD2 = torch::sub(hC1, hB1);
    auto hD1 = torch::relu(hD2);
  }

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(D, hD.cpu()), true);

  // this should not be cache Miss, check logs..
  // [toDo] this is failing need to fix it.
  {
    // graph 1
    torch::Tensor A = torch::randn({2, 2});
    torch::Tensor B = torch::randn({2, 2});
    auto hA = A.to(torch::kHPU);
    auto hB = B.to(torch::kHPU);
    auto hC = torch::add(hA, hB);
    auto hD = torch::relu(hC);
    // cpu output
    auto C = torch::add(A, B);
    auto D = torch::relu(C);

    HbLazyTensor::StepMarker({});
    EXPECT_EQ(allclose(D, hD.cpu()), true);
  }
}

TEST_F(LazyFwdRunningHashTest, remove_post_order_view_test) {
  torch::Tensor A = torch::randn({2, 3, 4, 5});
  auto B = A.view({-1, 5, 6});
  auto C = B.add(0.5);
  auto E = A.add(1.5);
  auto D = B.add(1.5);

  auto hA = A.to(torch::kHPU);
  auto hB = hA.view({-1, 5, 6});
  HbLazyTensor::StepMarker({});

  auto hC = hB.add(0.5);
  HbLazyTensor::StepMarker({});

  auto hE = hA.add(1.5);
  HbLazyTensor::StepMarker({});

  auto hD = hB.add(1.5);
  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(A, hA.cpu()), true);
}