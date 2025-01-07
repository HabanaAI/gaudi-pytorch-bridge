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
#define SIZE(...) __VA_ARGS__

#define HPU_CUMSUM_TEST(op, in_size, dimension, datatype)            \
  TEST_F(HpuOpTest, op) {                                            \
    torch::ScalarType dtype = datatype;                              \
    GenerateInputs(1, {in_size}, {dtype});                           \
    auto expected = torch::cumsum(GetCpuInput(0), dimension, dtype); \
    auto result = torch::cumsum(GetHpuInput(0), dimension, dtype);   \
    Compare(expected, result);                                       \
  }

class HpuOpTest : public HpuOpTestUtil {};
HPU_CUMSUM_TEST(cumsum_f32, SIZE({2, 3}), 1, torch::kFloat)
HPU_CUMSUM_TEST(cumsum_bf16, SIZE({16, 8, 4}), -3, torch::kBFloat16)
HPU_CUMSUM_TEST(cumsum_i32, SIZE({16, 4, 2, 3}), 3, torch::kInt32)

TEST_F(HpuOpTest, cumsum_without_dtype_attr) {
  torch::ScalarType dtype = torch::kInt32;
  GenerateInputs(1, {{3, 4, 8, 16}}, dtype);
  auto expected = torch::cumsum(GetCpuInput(0), -2);
  auto result = torch::cumsum(GetHpuInput(0), -2);
  Compare(expected, result);
}

TEST_F(HpuOpTest, cumsum_without_dtype_attr_2) {
  torch::ScalarType dtype = torch::kFloat;
  GenerateInputs(1, {{8, 24, 24, 3}}, dtype);
  auto expected = torch::cumsum(GetCpuInput(0), 2);
  auto result = torch::cumsum(GetHpuInput(0), 2);
  Compare(expected, result);
}

TEST_F(HpuOpTest, cumsum_without_dtype_attr_3) {
  torch::ScalarType dtype = torch::kBFloat16;
  GenerateInputs(1, {{8, 24, 24}}, dtype);
  auto expected = torch::cumsum(GetCpuInput(0), 0);
  auto result = torch::cumsum(GetHpuInput(0), 0);
  Compare(expected, result);
}
