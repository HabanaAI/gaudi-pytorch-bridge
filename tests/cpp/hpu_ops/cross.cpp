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

#define HPU_CROSS_OUT_TEST(op, in_size, dimension, datatype)                  \
  TEST_F(HpuOpTest, op) {                                                     \
    torch::ScalarType dtype = datatype;                                       \
    GenerateInputs(2, {in_size, in_size}, {dtype, dtype});                    \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::cross_outf(GetCpuInput(0), GetCpuInput(1), dimension, expected);   \
    torch::cross_outf(GetHpuInput(0), GetHpuInput(1), dimension, result);     \
    Compare(expected, result);                                                \
  }

#define HPU_CROSS_OUT_TEST_WITHOUT_DIM(op, in_size, datatype)                 \
  TEST_F(HpuOpTest, op) {                                                     \
    torch::ScalarType dtype = datatype;                                       \
    GenerateInputs(2, {in_size, in_size}, {dtype, dtype});                    \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::cross_out(expected, GetCpuInput(0), GetCpuInput(1));               \
    torch::cross_out(result, GetHpuInput(0), GetHpuInput(1));                 \
    Compare(expected, result);                                                \
  }

#define HPU_CROSS_TEST(op, in_size, dimension, dtype)                        \
  TEST_F(HpuOpTest, op) {                                                    \
    GenerateInputs(2, {in_size, in_size}, {dtype, dtype});                   \
    auto expected = torch::cross(GetCpuInput(0), GetCpuInput(1), dimension); \
    auto result = torch::cross(GetHpuInput(0), GetHpuInput(1), dimension);   \
    Compare(expected, result);                                               \
  }

#define HPU_CROSS_TEST_WITHOUT_DIM(op, in_size, dtype)            \
  TEST_F(HpuOpTest, op) {                                         \
    GenerateInputs(2, {{in_size}, {in_size}}, {dtype, dtype});    \
    auto expected = torch::cross(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::cross(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                    \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_CROSS_TEST(cross, SIZE({3}), 0, torch::kBFloat16)
HPU_CROSS_TEST(cross_2d, SIZE({4, 3}), 1, torch::kInt)
HPU_CROSS_TEST(cross_3d, SIZE({6, 4, 3}), -1, torch::kFloat)
HPU_CROSS_TEST(cross_4d, SIZE({6, 3, 4, 3}), -3, torch::kInt)
HPU_CROSS_TEST(cross_5d, SIZE({5, 6, 4, 3, 7}), 3, torch::kBFloat16)

HPU_CROSS_TEST_WITHOUT_DIM(cross_empty_dim, SIZE({3}), torch::kBFloat16)
HPU_CROSS_TEST_WITHOUT_DIM(cross_empty_dim2d, SIZE({3, 8}), torch::kInt)
HPU_CROSS_TEST_WITHOUT_DIM(cross_empty_dim3d, SIZE({5, 5, 3}), torch::kFloat)
HPU_CROSS_TEST_WITHOUT_DIM(
    cross_empty_dim4d,
    SIZE({7, 3, 5, 3}),
    torch::kBFloat16)
HPU_CROSS_TEST_WITHOUT_DIM(
    cross_empty_dim5d,
    SIZE({7, 7, 3, 5, 6}),
    torch::kFloat)

HPU_CROSS_OUT_TEST(cross_out_2d, SIZE({4, 3}), 1, torch::kFloat)
HPU_CROSS_OUT_TEST(cross_out_3d, SIZE({2, 4, 3}), -1, torch::kBFloat16)
HPU_CROSS_OUT_TEST(cross_out_4d, SIZE({5, 4, 3, 9}), 2, torch::kInt)
HPU_CROSS_OUT_TEST(cross_out_5d, SIZE({2, 3, 7, 8, 5}), -4, torch::kBFloat16)

HPU_CROSS_OUT_TEST_WITHOUT_DIM(cross_out_empty_dim2d, SIZE({4, 3}), torch::kInt)
HPU_CROSS_OUT_TEST_WITHOUT_DIM(
    cross_out_empty_dim3d,
    SIZE({2, 4, 3}),
    torch::kBFloat16)
HPU_CROSS_OUT_TEST_WITHOUT_DIM(
    cross_out_empty_dim4d,
    SIZE({5, 4, 3, 8}),
    torch::kBFloat16)
HPU_CROSS_OUT_TEST_WITHOUT_DIM(
    cross_out_empty_dim5d,
    SIZE({2, 4, 3, 7, 8}),
    torch::kInt)
