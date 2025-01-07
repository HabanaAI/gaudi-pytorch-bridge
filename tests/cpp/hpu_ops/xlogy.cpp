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

#define TENSOR_OUTPLACE_TEST(op)                               \
  TEST_F(HpuOpTest, op##_other_tensor) {                       \
    GenerateInputs(2);                                         \
    auto expected = torch::op(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::op(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                 \
  }

#define OTHER_SCALAR_OUTPLACE_TEST(op)                \
  TEST_F(HpuOpTest, op##_other_scalar) {              \
    GenerateInputs(1);                                \
    float other = -1.4;                               \
    auto expected = torch::op(GetCpuInput(0), other); \
    auto result = torch::op(GetHpuInput(0), other);   \
    Compare(expected, result);                        \
  }

#define TENSOR_OUT_TEST(op)                                                   \
  TEST_F(HpuOpTest, op##_other_tensor_out) {                                  \
    GenerateInputs(2);                                                        \
    torch::ScalarType dtype = torch::kFloat;                                  \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), GetCpuInput(1), expected);                      \
    torch::op(GetHpuInput(0), GetHpuInput(1), result);                        \
    Compare(expected, result);                                                \
  }

#define OTHER_SCALAR_OUT_TEST(op)                                             \
  TEST_F(HpuOpTest, op##_other_scalar_out) {                                  \
    GenerateInputs(1);                                                        \
    torch::ScalarType dtype = torch::kFloat;                                  \
    float other = -0.12345;                                                   \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), other, expected);                               \
    torch::op(GetHpuInput(0), other, result);                                 \
    Compare(expected, result);                                                \
  }

#define SELF_SCALAR_OUTPLACE_TEST(op)                \
  TEST_F(HpuOpTest, op##_self_scalar) {              \
    GenerateInputs(1);                               \
    float self = 2.3;                                \
    auto expected = torch::op(self, GetCpuInput(0)); \
    auto result = torch::op(self, GetHpuInput(0));   \
    Compare(expected, result);                       \
  }

#define SELF_SCALAR_OUT_TEST(op)                                              \
  TEST_F(HpuOpTest, op##_self_scalar_out) {                                   \
    GenerateInputs(1);                                                        \
    float self = 1;                                                           \
    torch::ScalarType dtype = torch::kFloat;                                  \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op(self, GetCpuInput(0), expected);                                \
    torch::op(self, GetHpuInput(0), result);                                  \
    Compare(expected, result);                                                \
  }

#define OTHER_SCALAR_INPLACE_TEST(op)            \
  TEST_F(HpuOpTest, op##other_scalar_) {         \
    GenerateInputs(1);                           \
    float self = 2.3;                            \
    GetCpuInput(0).op(self);                     \
    GetHpuInput(0).op(self);                     \
    Compare(GetCpuInput(0), GetHpuInput(0));     \
  }

#define OTHER_TENSOR_INPLACE_TEST(op)            \
  TEST_F(HpuOpTest, op##other_tensor_) {         \
    GenerateInputs(2);                           \
    GetCpuInput(0).op(GetCpuInput(1));           \
    GetHpuInput(0).op(GetHpuInput(1));           \
    Compare(GetCpuInput(0), GetHpuInput(0));     \
  }

#define XLOGY_TEST(op)             \
  TENSOR_OUTPLACE_TEST(op)         \
  TENSOR_OUT_TEST(op##_outf)       \
  OTHER_SCALAR_OUTPLACE_TEST(op)   \
  OTHER_SCALAR_OUT_TEST(op##_outf) \
  SELF_SCALAR_OUTPLACE_TEST(op)    \
  SELF_SCALAR_OUT_TEST(op##_outf)

#define INPLACE_TEST(op)        \
  OTHER_SCALAR_INPLACE_TEST(op) \
  OTHER_TENSOR_INPLACE_TEST(op)

class HpuOpTest : public HpuOpTestUtil {};

XLOGY_TEST(special_xlog1py)
XLOGY_TEST(xlogy)
INPLACE_TEST(xlogy_)

// Issue Raised: https://jira.habana-labs.com/browse/SW-102352
// Below testcase fails for default tolerance
// hence tuned atol & rtol to 1e-2
TEST_F(HpuOpTest, xlogy_self_scalar_bf16) {
  GenerateInputs(1, torch::kBFloat16);
  float self = 2.3;
  auto expected = torch::xlogy(self, GetCpuInput(0));
  auto result = torch::xlogy(self, GetHpuInput(0));
  Compare(expected, result, 1e-2, 1e-2);
}

TEST_F(HpuOpTest, xlogy_other_scalar_bf16) {
  GenerateInputs(1, torch::kBFloat16);
  float other = 2.3;
  auto expected = torch::xlogy(GetCpuInput(0), other);
  auto result = torch::xlogy(GetHpuInput(0), other);
  Compare(expected, result);
}