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

#define HPU_FLIP_TEST(test_name, in_size, datatype, dims) \
  TEST_F(HpuOpTest, test_name) {                          \
    torch::ScalarType dtype = datatype;                   \
    GenerateInputs(1, {in_size}, {dtype});                \
    auto expected = torch::flip(GetCpuInput(0), dims);    \
    auto result = torch::flip(GetHpuInput(0), dims);      \
    Compare(expected, result);                            \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_FLIP_TEST(flip_f32, SIZE({16, 8}), torch::kFloat, SIZE({0, 1}))
HPU_FLIP_TEST(flip_bf16, SIZE({16, 8, 8}), torch::kBFloat16, SIZE({2, 1}))
HPU_FLIP_TEST(flip_i8, SIZE({16, 8, 8, 2}), torch::kInt8, SIZE({-2, -3, -4}))
HPU_FLIP_TEST(
    flip_i32,
    SIZE({16, 8, 8, 2, 4}),
    torch::kInt32,
    SIZE({0, 4, 3, 2}))
HPU_FLIP_TEST(flip_u8, SIZE({4, 16, 2}), torch::kUInt8, SIZE({-3, -1, -2}))
