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

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, eq_scalar) {
  GenerateInputs(1, {{2, 3, 4}});
  float other = 1.1;

  GetCpuInput(0).eq_(other);
  GetHpuInput(0).eq_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, eq_tensor) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  GetCpuInput(0).eq_(GetCpuInput(1));
  GetHpuInput(0).eq_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, equal) {
  auto tensor1 = torch::tensor({1, 0, 0}, "hpu");
  auto tensor2 = torch::tensor({1, 1, 1}, "hpu");
  EXPECT_FALSE(torch::equal(tensor1, tensor2));
}

TEST_F(HpuOpTest, equal_bf16) {
  auto ones = torch::ones(
      {2, 3, 4}, torch::TensorOptions(torch::kBFloat16).device("hpu"));
  EXPECT_TRUE(torch::equal(ones, ones));
}

TEST_F(HpuOpTest, equal_diff_shape) {
  auto ones = torch::ones({2, 12, 8, 9}, "hpu");
  auto zeros = torch::zeros({1, 122, 3, 4, 6}, "hpu");
  EXPECT_FALSE(torch::equal(ones, zeros));
}

TEST_F(HpuOpTest, equal_f32) {
  auto ones = torch::ones({12}, "hpu");
  auto zeros = torch::zeros({12}, "hpu");
  EXPECT_FALSE(torch::equal(ones, zeros));
}

TEST_F(HpuOpTest, equal_i8) {
  auto ones =
      torch::ones({8, 4, 4}, torch::TensorOptions(torch::kInt8).device("hpu"));
  auto zeros =
      torch::zeros({8, 4, 4}, torch::TensorOptions(torch::kInt8).device("hpu"));
  EXPECT_FALSE(torch::equal(ones, zeros));
}

TEST_F(HpuOpTest, equal_i32) {
  auto ones =
      torch::ones({2, 3, 4}, torch::TensorOptions(torch::kInt32).device("hpu"));
  EXPECT_TRUE(torch::equal(ones, ones));
}

TEST_F(HpuOpTest, equal_u8) {
  auto ones = torch::ones(
      {8, 4, 4, 3}, torch::TensorOptions(torch::kUInt8).device("hpu"));
  EXPECT_TRUE(torch::equal(ones, ones));
}
