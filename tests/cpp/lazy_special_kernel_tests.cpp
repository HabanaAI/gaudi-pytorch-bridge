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

using namespace habana_lazy;
using namespace at;

#define MAX_VALUE_TO_TEST 127
#define MIN_VALUE_TO_TEST -127

#define HPU_LAZY_KERNEL_TEST(op_code, min_val, max_val)                        \
  TEST_F(LazySpecialKernelTest, op_code##Forward) {                            \
    auto A = torch::randn(4);                                                  \
    auto min = min_val;                                                        \
    auto max = max_val;                                                        \
    A = at::clamp(A, min, max);                                                \
    auto hA = A.to(torch::kHPU);                                               \
    auto expectedOutput = torch::op_code(A);                                   \
    auto habanaOutput = torch::op_code(hA);                                    \
    EXPECT_EQ(                                                                 \
        allclose(habanaOutput.to("cpu"), expectedOutput, 0.001, 0.001), true); \
  }

class LazySpecialKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazySpecialKernelTest, AsinForward) {
  auto A = torch::randn(4, torch::dtype(torch::kFloat));
  auto min = -1.0;
  auto max = 1.0;
  A = at::clamp(A, min, max);
  auto hA = A.to(torch::kHPU);
  auto expectedOutput = torch::asin(A);
  auto habanaOutput = torch::asin(hA);
  EXPECT_EQ(
      allclose(habanaOutput.to("cpu"), expectedOutput, 0.001, 0.001), true);
}

HPU_LAZY_KERNEL_TEST(acos, -1.0, 1.0)
HPU_LAZY_KERNEL_TEST(acosh, 1.0, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(asinh, MIN_VALUE_TO_TEST, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(atan, MIN_VALUE_TO_TEST, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(atanh, -1.0, 1.0)
HPU_LAZY_KERNEL_TEST(cosh, MIN_VALUE_TO_TEST, MAX_VALUE_TO_TEST)

HPU_LAZY_KERNEL_TEST(acos_, -1.0, 1.0)
HPU_LAZY_KERNEL_TEST(acosh_, 1.0, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(asinh_, MIN_VALUE_TO_TEST, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(atan_, MIN_VALUE_TO_TEST, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(tanh_, MIN_VALUE_TO_TEST, MAX_VALUE_TO_TEST)
HPU_LAZY_KERNEL_TEST(atanh_, -1.0, 1.0)
HPU_LAZY_KERNEL_TEST(cos_, -1.0, 1.0)
HPU_LAZY_KERNEL_TEST(cosh_, -1.0, 1.0)
