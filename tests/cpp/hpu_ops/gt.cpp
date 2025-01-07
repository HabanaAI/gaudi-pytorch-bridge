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

#include <gtest/gtest-param-test.h>
#include "habana_kernels/fallback_helper.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, gt_scalar) {
  GenerateInputs(1);
  int other = 0;

  GetCpuInput(0).gt_(other);
  GetHpuInput(0).gt_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, gt_tensor) {
  GenerateInputs(2);
  int other = 1;

  GetCpuInput(0).gt_(GetCpuInput(1));
  GetHpuInput(0).gt_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

class GTDTypeSupportTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_P(GTDTypeSupportTest, GTScalarOut) {
  auto dtype = GetParam();
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kHPU);
  auto input = torch::tensor({1, 2, 3, 4}, options);
  auto output = torch::empty({}, options);

  torch::gt_out(output, input, 2);

  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten::gt.Scalar_out"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    GTScalarOutFallback,
    GTDTypeSupportTest,
    testing::Values(
        torch::kBFloat16,
        torch::kFloat32,
        torch::kInt32,
        torch::kUInt8,
        torch::kInt8,
        torch::kInt64));
