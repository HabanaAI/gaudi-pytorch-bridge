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

#define INDEX_SELECT_OUT_TEST(                                             \
    test_name, in_size, max_value, datatype, index_value, dim, out_size)   \
  TEST_F(HpuOpTest, test_name) {                                           \
    torch::ScalarType dtype = datatype;                                    \
    GenerateIntInputs(1, {index_value}, 0, max_value);                     \
    auto cpu_index = GetCpuInput(0).to(torch::kLong);                      \
    auto hpu_index = cpu_index.to(torch::kHPU);                            \
    auto expected = torch::empty(out_size, dtype);                         \
    auto result =                                                          \
        torch::empty(out_size, torch::TensorOptions(dtype).device("hpu")); \
    GenerateInputs(1, {in_size}, {dtype});                                 \
    torch::index_select_outf(GetCpuInput(0), dim, cpu_index, expected);    \
    torch::index_select_outf(GetHpuInput(0), dim, hpu_index, result);      \
    Compare(expected, result, 0, 0);                                       \
  }

#define INDEX_SELECT_TEST(                                               \
    test_name, in_size, max_value, datatype, index_value, dim, out_size) \
  TEST_F(HpuOpTest, test_name) {                                         \
    torch::ScalarType dtype = datatype;                                  \
    GenerateIntInputs(1, {index_value}, 0, max_value);                   \
    auto cpu_index = GetCpuInput(0).to(torch::kLong);                    \
    auto hpu_index = cpu_index.to(torch::kHPU);                          \
    GenerateInputs(1, {in_size}, {dtype});                               \
    auto expected = torch::index_select(GetCpuInput(0), dim, cpu_index); \
    auto result = torch::index_select(GetHpuInput(0), dim, hpu_index);   \
    Compare(expected, result, 0, 0);                                     \
  }

class HpuOpTest : public HpuOpTestUtil {};

INDEX_SELECT_TEST(index_select_1D, SIZE({10}), 10, torch::kInt, 4, 0, 0)
INDEX_SELECT_TEST(index_select_2D, SIZE({28, 28}), 28, torch::kInt, 5, 1, 0)

INDEX_SELECT_OUT_TEST(
    index_select_out_1D,
    SIZE({1024}),
    1024,
    torch::kInt,
    4,
    0,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_2D,
    SIZE({28, 28}),
    28,
    torch::kInt,
    5,
    1,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_3D,
    SIZE({8, 512, 512}),
    512,
    torch::kFloat,
    4,
    2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_4D,
    SIZE({8, 24, 24, 3}),
    24,
    torch::kChar,
    3,
    2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_5D,
    SIZE({8, 12, 12, 16, 24}),
    24,
    torch::kBFloat16,
    4,
    4,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_int,
    SIZE({128, 256, 200}),
    200,
    torch::kInt,
    4,
    2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_char,
    SIZE({200, 356, 20}),
    200,
    torch::kChar,
    4,
    0,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_neg_float,
    SIZE({8, 24, 24, 16, 36}),
    16,
    torch::kFloat,
    2,
    -2,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_neg_bfloat,
    SIZE({8, 24, 24, 16, 36}),
    24,
    torch::kBFloat16,
    2,
    -3,
    0)
INDEX_SELECT_OUT_TEST(
    index_select_out_neg_char,
    SIZE({8, 12, 12, 16, 36}),
    36,
    torch::kChar,
    2,
    -1,
    0)

INDEX_SELECT_OUT_TEST(
    index_select_out_shape,
    SIZE({8, 24, 32, 3}),
    32,
    torch::kChar,
    2,
    2,
    32)