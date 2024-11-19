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

class LogicalOutHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<c10::ScalarType> {};

#define HPU_LOGICAL_OUT_TEST(op)                                               \
  TEST_P(LogicalOutHpuOpTest, op) {                                            \
    const auto& dtype = GetParam();                                            \
    GenerateInputs(2, dtype);                                                  \
    torch::ScalarType dtypef = torch::kFloat;                                  \
    auto expected = torch::empty(0, dtypef);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtypef).device("hpu")); \
    torch::op(GetCpuInput(0), GetCpuInput(1), expected);                       \
    torch::op(GetHpuInput(0), GetHpuInput(1), result);                         \
    Compare(expected, result);                                                 \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      op,                                                                      \
      LogicalOutHpuOpTest,                                                     \
      testing::Values(                                                         \
          torch::kFloat, torch::kBFloat16, torch::kByte, torch::kChar));

class LogicalNotHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<c10::ScalarType> {};

#define HPU_LOGICAL_NOT_OUT_TEST(op)                                           \
  TEST_P(LogicalNotHpuOpTest, op) {                                            \
    const auto& dtype = GetParam();                                            \
    GenerateInputs(1, dtype);                                                  \
    torch::ScalarType dtypef = torch::kBool;                                   \
    auto expected = torch::empty(0, dtypef);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtypef).device("hpu")); \
    torch::op(GetCpuInput(0), expected);                                       \
    torch::op(GetHpuInput(0), result);                                         \
    Compare(expected, result);                                                 \
  }                                                                            \
  INSTANTIATE_TEST_SUITE_P(                                                    \
      op, LogicalNotHpuOpTest, testing::Values(torch::kByte, torch::kChar));

class LogicalHpuOpTest : public HpuOpTestUtil,
                         public testing::WithParamInterface<c10::ScalarType> {};

#define HPU_LOGICAL_TEST(op)                                   \
  TEST_P(LogicalHpuOpTest, op) {                               \
    const auto& dtype = GetParam();                            \
    GenerateInputs(2, dtype);                                  \
    auto expected = torch::op(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::op(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                 \
  }                                                            \
  INSTANTIATE_TEST_SUITE_P(                                    \
      op,                                                      \
      LogicalHpuOpTest,                                        \
      testing::Values(                                         \
          torch::kFloat, torch::kBFloat16, torch::kByte, torch::kChar));

class LogicalInplaceHpuOpTest
    : public HpuOpTestUtil,
      public testing::WithParamInterface<c10::ScalarType> {};
#define HPU_LOGICAL_INPLACE_TEST(op)         \
  TEST_P(LogicalInplaceHpuOpTest, op) {      \
    const auto& dtype = GetParam();          \
    GenerateInputs(2, dtype);                \
    GetCpuInput(0).op(GetCpuInput(1));       \
    GetHpuInput(0).op(GetHpuInput(1));       \
    Compare(GetCpuInput(0), GetHpuInput(0)); \
  }                                          \
  INSTANTIATE_TEST_SUITE_P(                  \
      op,                                    \
      LogicalInplaceHpuOpTest,               \
      testing::Values(                       \
          torch::kFloat, torch::kBFloat16, torch::kByte, torch::kChar));

#define TEST_HPU_LOGICAL_OP(op)   \
  HPU_LOGICAL_OUT_TEST(op##_outf) \
  HPU_LOGICAL_TEST(op)            \
  HPU_LOGICAL_INPLACE_TEST(op##_)

#define TEST_HPU_LOGICAL_NOT_OP(op) HPU_LOGICAL_NOT_OUT_TEST(op##_outf)

TEST_HPU_LOGICAL_OP(logical_and)
TEST_HPU_LOGICAL_OP(logical_or)
TEST_HPU_LOGICAL_OP(logical_xor)
TEST_HPU_LOGICAL_NOT_OP(logical_not)
