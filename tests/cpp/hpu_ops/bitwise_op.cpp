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

#define MIN_VALUE -127
#define MAX_VALUE 128

#define HPU_BITWISE_USUAL_TEST(op_code, dtype)                              \
  TEST_F(BitwiseHpuOpTest, op_code##Usual) {                                \
    auto input = torch::randint(MIN_VALUE, MAX_VALUE, {2, 3, 4}).to(dtype); \
    auto other = torch::randint(MIN_VALUE, MAX_VALUE, {3, 4}).to(dtype);    \
    auto hinput = input.to(torch::kHPU);                                    \
    auto hother = other.to(torch::kHPU);                                    \
    auto expected = torch::op_code(input, other);                           \
    auto result = torch::op_code(hinput, hother);                           \
    Compare(expected, result);                                              \
  }

#define HPU_BITWISE_INPLACE_TEST(op_code, dtype)                            \
  TEST_F(BitwiseHpuOpTest, op_code##Inplace) {                              \
    auto input = torch::randint(MIN_VALUE, MAX_VALUE, {2, 3, 4}).to(dtype); \
    auto other = torch::randint(MIN_VALUE, MAX_VALUE, {2, 3, 4}).to(dtype); \
    auto hinput = input.to(torch::kHPU);                                    \
    auto hother = other.to(torch::kHPU);                                    \
    input.op_code(other);                                                   \
    hinput.op_code(hother);                                                 \
    Compare(input, hinput);                                                 \
  }

class BitwiseHpuOpTest : public HpuOpTestUtil {};

HPU_BITWISE_USUAL_TEST(bitwise_and, torch::kInt)
HPU_BITWISE_USUAL_TEST(bitwise_or, torch::kBool)
HPU_BITWISE_USUAL_TEST(bitwise_xor, torch::kInt)

HPU_BITWISE_INPLACE_TEST(bitwise_and_, torch::kBool)
HPU_BITWISE_INPLACE_TEST(bitwise_or_, torch::kInt)
HPU_BITWISE_INPLACE_TEST(bitwise_xor_, torch::kBool)