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
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

#include <cstdlib>

using namespace habana_lazy;
using namespace at;

class LazyBitwiseKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyBitwiseKernelTest, BitwiseAndTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor B = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out = torch::bitwise_and(hA, hB);

  torch::Tensor out_cpu = torch::bitwise_and(A, B);
  torch::Tensor out_h = out.to(torch::kCPU);
  EXPECT_EQ(allclose(out_h.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}

TEST_F(LazyBitwiseKernelTest, BitwiseXorTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor B = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out = torch::bitwise_xor(hA, hB);

  torch::Tensor out_cpu = torch::bitwise_xor(A, B);
  torch::Tensor out_h = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_h.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}

TEST_F(LazyBitwiseKernelTest, BitwiseNotTest) {
  torch::Tensor A = torch::randint(-10, 10, {3, 2}) > 0;
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor out = torch::bitwise_not(hA);

  torch::Tensor out_cpu = torch::bitwise_not(A);
  torch::Tensor out_h = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_h.to(torch::kI8), out_cpu.to(torch::kI8)), true);
}
