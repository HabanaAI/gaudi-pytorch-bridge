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

#define TENSOR_TYPE_bool torch::kBool
#define TENSOR_TYPE_int torch::kInt

#define GET_TENSOR_TYPE(type) TENSOR_TYPE_##type

#define HPU_BITWISE_OUTPLACE_TEST(op, dtype)                        \
  TEST_F(HpuOpTest, op) {                                           \
    /* Tensor Tensor inputs */                                      \
    GenerateInputs(2, {{4, 5}, {3, 4, 5}}, GET_TENSOR_TYPE(dtype)); \
    auto expected1 = torch::op(GetCpuInput(0), GetCpuInput(1));     \
    auto result1 = torch::op(GetHpuInput(0), GetHpuInput(1));       \
    Compare(expected1, result1);                                    \
    /* Tensor Scalar inputs */                                      \
    dtype s = GenerateScalar<dtype>();                              \
    auto expected2 = torch::op(GetCpuInput(0), s);                  \
    auto result2 = torch::op(GetHpuInput(0), s);                    \
    Compare(expected2, result2);                                    \
  }

#define HPU_BITWISE_INPLACE_TEST(op, dtype)                         \
  TEST_F(HpuOpTest, op) {                                           \
    /* Tensor Tensor inputs */                                      \
    GenerateInputs(2, {{3, 4, 5}, {4, 5}}, GET_TENSOR_TYPE(dtype)); \
    GetCpuInput(0).op(GetCpuInput(1));                              \
    GetHpuInput(0).op(GetHpuInput(1));                              \
    Compare(GetCpuInput(0), GetHpuInput(0));                        \
    /* Tensor Scalar inputs */                                      \
    dtype s = GenerateScalar<dtype>();                              \
    GetCpuInput(0).op(s);                                           \
    GetHpuInput(0).op(s);                                           \
    Compare(GetCpuInput(0), GetHpuInput(0));                        \
  }

#define HPU_BITWISE_OUT_TEST(op, dtype_)                                \
  TEST_F(HpuOpTest, op) {                                               \
    /* Tensor Tensor inputs */                                          \
    auto dtype = GET_TENSOR_TYPE(dtype_);                               \
    GenerateInputs(2, {{3, 4, 5}, {4, 5}}, dtype);                      \
    auto exp1 = torch::empty(0, dtype);                                 \
    auto res1 =                                                         \
        torch::empty(0, torch::TensorOptions(dtype).device(c10::kHPU)); \
    torch::op(GetCpuInput(0), GetCpuInput(1), exp1);                    \
    torch::op(GetHpuInput(0), GetHpuInput(1), res1);                    \
    Compare(exp1, res1);                                                \
    /* Tensor Scalar inputs */                                          \
    dtype_ s = GenerateScalar<dtype_>();                                \
    auto exp2 = torch::empty(0, dtype);                                 \
    auto res2 =                                                         \
        torch::empty(0, torch::TensorOptions(dtype).device(c10::kHPU)); \
    torch::op(GetCpuInput(0), s, exp2);                                 \
    torch::op(GetHpuInput(0), s, res2);                                 \
    Compare(exp2, res2);                                                \
  }

class HpuOpTest : public HpuOpTestUtil {};

#define TEST_HPU_BITWISE_OP(op)         \
  HPU_BITWISE_OUTPLACE_TEST(op, int)    \
  HPU_BITWISE_INPLACE_TEST(op##_, bool) \
  HPU_BITWISE_OUT_TEST(op##_outf, int)

TEST_HPU_BITWISE_OP(bitwise_and)
TEST_HPU_BITWISE_OP(bitwise_or)
TEST_HPU_BITWISE_OP(bitwise_xor)

// bitwise_not takes only one input, and cannot use the macro above which is
// generalized for two inputs
TEST_F(HpuOpTest, bitwise_not) {
  GenerateIntInputs(1, {{4, 5, 6}}, -10000, 10000);
  auto exp = torch::bitwise_not(GetCpuInput(0));
  auto res = torch::bitwise_not(GetHpuInput(0));

  Compare(exp, res);
}

TEST_F(HpuOpTest, bitwise_not_) {
  GenerateInputs(1, {10}, {torch::kBool});
  auto exp = GetCpuInput(0).bitwise_not_();
  auto res = GetHpuInput(0).bitwise_not_();

  Compare(exp, res);
}

TEST_F(HpuOpTest, bitwise_not_out) {
  GenerateIntInputs(1, {{4, 5, 6}}, -10000, 10000);
  auto exp = torch::empty(0, torch::kInt);
  auto res =
      torch::empty(0, torch::TensorOptions(torch::kInt).device(c10::kHPU));
  torch::bitwise_not_outf(GetCpuInput(0), exp);
  torch::bitwise_not_outf(GetHpuInput(0), res);

  Compare(exp, res);
}
