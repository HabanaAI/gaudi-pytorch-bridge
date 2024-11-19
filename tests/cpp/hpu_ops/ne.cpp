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
#include <cstdint>
#include <unordered_map>
#include "habana_kernels/fallback_helper.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

class NeDtypeSupportTest : public DTypeSupportTest<c10::ScalarType> {};

TEST_F(HpuOpTest, ne_scalar_out) {
  GenerateInputs(1, torch::kFloat);
  float compVal = -1.1f;
  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::ne_outf(GetCpuInput(0), compVal, expected);
  torch::ne_outf(GetHpuInput(0), compVal, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, ne_tensor_out) {
  GenerateInputs(2, {{1, 2, 1}, {2, 2, 1}}, {torch::kFloat, torch::kBFloat16});

  torch::ScalarType dtype = torch::kBool;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::ne_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::ne_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, ne_scalar_inplace) {
  GenerateInputs(1, {{2, 64, 24, 12, 2}}, {torch::kInt});
  float other = 2.5;

  GetCpuInput(0).ne_(other);
  GetHpuInput(0).ne_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, ne_tensor_inplace) {
  GenerateInputs(2);

  GetCpuInput(0).ne_(GetCpuInput(1));
  GetHpuInput(0).ne_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_P(NeDtypeSupportTest, NeScalarOutTest) {
  auto dtype = GetParam();
  auto compare_value = at::Scalar(static_cast<int64_t>(10));
  auto options = torch::TensorOptions().dtype(dtype).device(torch::kHPU);
  auto input_tensor = torch::tensor({10, -10}, options);
  auto output_tensor = torch::empty({2}, options.dtype(torch::kBool));

  auto out_cpu =
      torch::ne_out(input_tensor, output_tensor, compare_value).to(torch::kCPU);

  const auto& op_fallback_frequency =
      habana::HpuFallbackHelper::get()->get_op_count();
  EXPECT_EQ(
      op_fallback_frequency.find("aten::ne.Scalar_out"),
      op_fallback_frequency.end());
}

INSTANTIATE_TEST_SUITE_P(
    TypeSupportTest,
    NeDtypeSupportTest,
    testing::Values(
        torch::kBFloat16,
        torch::kFloat32,
        torch::kInt32,
        torch::kInt64,
        torch::kInt8,
        torch::kInt16));
