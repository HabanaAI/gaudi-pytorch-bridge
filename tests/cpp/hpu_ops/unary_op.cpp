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

#define HPU_UNARY_USUAL_TEST(op)               \
  TEST_F(HpuOpTest, op) {                      \
    GenerateInputs(1, {{2, 3}});               \
    auto expected = torch::op(GetCpuInput(0)); \
    auto result = torch::op(GetHpuInput(0));   \
    Compare(expected, result);                 \
  }

#define HPU_UNARY_INPLACE_TEST(op)  \
  TEST_F(HpuOpTest, op) {           \
    GenerateInputs(1, {{2, 3, 2}}); \
    auto expected = GetCpuInput(0); \
    auto result = GetHpuInput(0);   \
    expected = torch::op(expected); \
    result = torch::op(result);     \
    Compare(expected, result);      \
  }

#define HPU_UNARY_OUT_TEST(op)                                                \
  TEST_F(HpuOpTest, op) {                                                     \
    GenerateInputs(1, {{1, 2, 3, 2}});                                        \
    torch::ScalarType dtype = torch::kFloat;                                  \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), expected);                                      \
    torch::op(GetHpuInput(0), result);                                        \
    Compare(expected, result);                                                \
  }

class HpuOpTest : public HpuOpTestUtil {};

HPU_UNARY_USUAL_TEST(arccos)
HPU_UNARY_USUAL_TEST(arccosh)
HPU_UNARY_USUAL_TEST(arcsin)
HPU_UNARY_USUAL_TEST(arcsinh)
HPU_UNARY_USUAL_TEST(arctan)
HPU_UNARY_USUAL_TEST(arctanh)
HPU_UNARY_USUAL_TEST(sinh)
HPU_UNARY_USUAL_TEST(tan)

HPU_UNARY_INPLACE_TEST(arccos_)
HPU_UNARY_INPLACE_TEST(arccosh_)
HPU_UNARY_INPLACE_TEST(arcsin_)
HPU_UNARY_INPLACE_TEST(arcsinh_)
HPU_UNARY_INPLACE_TEST(arctan_)
HPU_UNARY_INPLACE_TEST(arctanh_)
HPU_UNARY_INPLACE_TEST(sinh_)
HPU_UNARY_INPLACE_TEST(tan_)

HPU_UNARY_OUT_TEST(arccos_outf)
HPU_UNARY_OUT_TEST(arccosh_outf)
HPU_UNARY_OUT_TEST(arcsin_outf)
HPU_UNARY_OUT_TEST(arcsinh_outf)
HPU_UNARY_OUT_TEST(arctan_outf)
HPU_UNARY_OUT_TEST(arctanh_outf)
HPU_UNARY_OUT_TEST(sinh_outf)
HPU_UNARY_OUT_TEST(tan_outf)