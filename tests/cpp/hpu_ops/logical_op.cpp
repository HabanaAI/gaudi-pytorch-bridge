/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class LogicalHpuOpTest : public HpuOpTestUtil,
                         public testing::WithParamInterface<c10::ScalarType> {};

#define HPU_LOGICAL_OUT_TEST(op)                                               \
  TEST_P(LogicalHpuOpTest, op) {                                               \
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
      LogicalHpuOpTest,                                                        \
      testing::Values(                                                         \
          torch::kFloat, torch::kBFloat16, torch::kByte, torch::kChar));

#define TEST_HPU_LOGICAL_OP(op) HPU_LOGICAL_OUT_TEST(op##_outf)

TEST_HPU_LOGICAL_OP(logical_and)
TEST_HPU_LOGICAL_OP(logical_or)
TEST_HPU_LOGICAL_OP(logical_xor)