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

#define TENSOR_TYPE_float torch::kFloat
#define TENSOR_TYPE_bfloat16 torch::kBFloat16
#define TENSOR_TYPE_int32 torch::kInt32

#define GET_TENSOR_TYPE(type) TENSOR_TYPE_##type

#define HPU_DOT_OUT_TEST(type)                                                \
  TEST_F(HpuOpTest, dot_out_##type) {                                         \
    torch::ScalarType dtype = GET_TENSOR_TYPE(type);                          \
    GenerateInputs(2, {{10}, {10}}, dtype);                                   \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::dot_outf(GetCpuInput(0), GetCpuInput(1), expected);                \
    torch::dot_outf(GetHpuInput(0), GetHpuInput(1), result);                  \
    Compare(expected, result);                                                \
  }

#define HPU_DOT_TEST(name, type, size)                          \
  TEST_F(HpuOpTest, dot_##name) {                               \
    torch::ScalarType dtype = GET_TENSOR_TYPE(type);            \
    GenerateInputs(2, {{size}, {size}}, dtype);                 \
    auto expected = torch::dot(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::dot(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                  \
  }

class HpuOpTest : public HpuOpTestUtil {};
HPU_DOT_OUT_TEST(float);
HPU_DOT_OUT_TEST(bfloat16);
HPU_DOT_OUT_TEST(int32);
HPU_DOT_TEST(float_1, float, 5);
HPU_DOT_TEST(bfloat16_1, bfloat16, 5);
HPU_DOT_TEST(float_2, float, 10);
HPU_DOT_TEST(bfloat16_2, bfloat16, 12);
HPU_DOT_TEST(int32, int32, 10);
