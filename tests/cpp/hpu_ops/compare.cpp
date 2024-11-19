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

#define TENSOR_OUTPLACE_TEST(op, dtype1, dtype2)                 \
  TEST_F(HpuOpTest, op##_tensor) {                               \
    GenerateInputs(2, {{3, 2, 1}, {3, 1, 1}}, {dtype1, dtype2}); \
    auto expected = torch::op(GetCpuInput(0), GetCpuInput(1));   \
    auto result = torch::op(GetHpuInput(0), GetHpuInput(1));     \
    Compare(expected, result);                                   \
  }

#define SCALAR_OUTPLACE_TEST(op, dtype1)              \
  TEST_F(HpuOpTest, op##_scalar) {                    \
    GenerateInputs(1, {{3, 2, 2}}, {dtype1});         \
    float other = 2.4;                                \
    auto expected = torch::op(GetCpuInput(0), other); \
    auto result = torch::op(GetHpuInput(0), other);   \
    Compare(expected, result);                        \
  }

#define TENSOR_OUT_TEST(op, dtype1, dtype2)                                   \
  TEST_F(HpuOpTest, op##_tensor) {                                            \
    GenerateInputs(2, {{4, 3, 2}, {4, 1, 2}}, {dtype1, dtype2});              \
    at::ScalarType dtype = at::promote_types(dtype1, dtype2);                 \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), GetCpuInput(1), expected);                      \
    torch::op(GetHpuInput(0), GetHpuInput(1), result);                        \
    Compare(expected, result);                                                \
  }

#define SCALAR_OUT_TEST(op, dtype1)                                            \
  TEST_F(HpuOpTest, op##_scalar) {                                             \
    GenerateInputs(1, {{4, 3, 2}}, {dtype1});                                  \
    int other = 2;                                                             \
    auto expected = torch::empty(0, dtype1);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype1).device("hpu")); \
    torch::op(GetCpuInput(0), other, expected);                                \
    torch::op(GetHpuInput(0), other, result);                                  \
    Compare(expected, result);                                                 \
  }

#define TENSOR_INPLACE_TEST(op, dtype1, dtype2)                  \
  TEST_F(HpuOpTest, op##tensor_inplace) {                        \
    GenerateInputs(2, {{3, 2, 2}, {3, 2, 1}}, {dtype1, dtype2}); \
    GetCpuInput(0).op(GetCpuInput(1));                           \
    GetHpuInput(0).op(GetHpuInput(1));                           \
    Compare(GetCpuInput(0), GetHpuInput(0));                     \
  }

#define SCALAR_INPLACE_TEST(op, dtype1)       \
  TEST_F(HpuOpTest, op##scalar_inplace) {     \
    GenerateInputs(1, {{4, 3, 2}}, {dtype1}); \
    float other = 1.6;                        \
    GetCpuInput(0).op(other);                 \
    GetHpuInput(0).op(other);                 \
    Compare(GetCpuInput(0), GetHpuInput(0));  \
  }

#define COMPARE_TEST(op)                                      \
  TENSOR_OUTPLACE_TEST(op, torch::kFloat, torch::kInt)        \
  TENSOR_OUT_TEST(op##_outf, torch::kFloat, torch::kBFloat16) \
  TENSOR_INPLACE_TEST(op##_, torch::kFloat, torch::kFloat)    \
  SCALAR_OUTPLACE_TEST(op, torch::kBFloat16)                  \
  SCALAR_OUT_TEST(op##_outf, torch::kFloat)                   \
  SCALAR_INPLACE_TEST(op##_, torch::kFloat)

class HpuOpTest : public HpuOpTestUtil {};

COMPARE_TEST(less)
COMPARE_TEST(less_equal)
COMPARE_TEST(greater)
COMPARE_TEST(greater_equal)
