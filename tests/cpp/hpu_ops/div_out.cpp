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

#define TENSOR_TYPE_float torch::kFloat32
#define TENSOR_TYPE_bfloat16 torch::kBFloat16
#define TENSOR_TYPE_int torch::kInt
#define TENSOR_TYPE_int8 torch::kInt8

#define GET_TENSOR_TYPE(type) TENSOR_TYPE_##type

class DivideHpuOpTest : public HpuOpTestUtil {};
// Test cases for all dtypes with broadcast test cases included
#define HPU_DIVIDE_OUT_TENSOR_TEST(type)                                \
  TEST_F(DivideHpuOpTest, div_out_##type) {                             \
    auto dtype = GET_TENSOR_TYPE(type);                                 \
    GenerateInputs(2, {{2, 4, 6}, {1, 4, 1}}, dtype);                   \
    auto result_dtype =                                                 \
        (dtype == torch::kBFloat16) ? torch::kBFloat16 : torch::kFloat; \
    auto expected = torch::empty(0, result_dtype);                      \
    auto result = expected.to(torch::kHPU);                             \
    torch::div_outf(GetCpuInput(0), GetCpuInput(1), expected);          \
    torch::div_outf(GetHpuInput(0), GetHpuInput(1), result);            \
    if (dtype == torch::kBFloat16) {                                    \
      /*TPC Kernel's precision, slightly differs fro CPU version for    \
       * bfloat16*/                                                     \
      /*Hence increased tolerance */                                    \
      Compare(expected, result, 0.1, 0.1);                              \
    } else {                                                            \
      Compare(expected, result);                                        \
    }                                                                   \
  }

HPU_DIVIDE_OUT_TENSOR_TEST(float);
HPU_DIVIDE_OUT_TENSOR_TEST(bfloat16);
HPU_DIVIDE_OUT_TENSOR_TEST(int);
HPU_DIVIDE_OUT_TENSOR_TEST(int8);

// Test cases for all dtypes with broadcast test cases included
#define HPU_DIVIDE_OUT_SCALAR_TEST(type)                                \
  TEST_F(DivideHpuOpTest, div_scalar_out_##type) {                      \
    auto dtype = GET_TENSOR_TYPE(type);                                 \
    GenerateInputs(1, dtype);                                           \
    auto result_dtype =                                                 \
        (dtype == torch::kBFloat16) ? torch::kBFloat16 : torch::kFloat; \
    auto other = GenerateScalar<int>();                                 \
    auto expected = torch::div(GetCpuInput(0), other);                  \
    auto result = torch::empty(0, result_dtype).to(torch::kHPU);        \
                                                                        \
    torch::div_outf(GetHpuInput(0), other, result);                     \
    Compare(expected, result);                                          \
  }

HPU_DIVIDE_OUT_SCALAR_TEST(float);
HPU_DIVIDE_OUT_SCALAR_TEST(bfloat16);
HPU_DIVIDE_OUT_SCALAR_TEST(int);
HPU_DIVIDE_OUT_SCALAR_TEST(int8);

class DivideScalarModeOutHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<
          std::tuple<c10::ScalarType, c10::optional<c10::string_view>>> {};

// Test cases which include both (tensor|scalar) with type_promotion
TEST_P(DivideScalarModeOutHpuOpTest, div_out_scalar_mode) {
  const auto& testParams = GetParam();
  const auto dtype = std::get<0>(testParams);
  const auto mode = std::get<1>(testParams);
  auto result_dtype = (mode == c10::nullopt && !(dtype == torch::kBFloat16))
      ? torch::kFloat
      : dtype;
  GenerateInputs(1, dtype);
  auto other = GenerateScalar<int>();

  auto exp = torch::div(GetCpuInput(0), other, mode);
  auto res = torch::empty(0, result_dtype).to(torch::kHPU);
  div_outf(GetHpuInput(0), other, mode, res);

  Compare(exp, res);
}

INSTANTIATE_TEST_SUITE_P(
    div_scalar_mode_out,
    DivideScalarModeOutHpuOpTest,
    ::testing::Combine(
        ::testing::Values<c10::ScalarType>(
            torch::kFloat,
            torch::kBFloat16,
            torch::kInt,
            torch::kInt8),
        ::testing::Values<c10::optional<c10::string_view>>(
            "floor",
            "trunc",
            c10::nullopt)));
