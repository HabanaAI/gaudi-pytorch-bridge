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

#define TENSOR_OUTPLACE_TEST(op)                               \
  TEST_F(HpuOpTest, op##_other_tensor) {                       \
    GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});                 \
    auto expected = torch::op(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::op(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                 \
  }

#define OTHER_SCALAR_OUTPLACE_TEST(op)                \
  TEST_F(HpuOpTest, op##_other_scalar) {              \
    GenerateInputs(1, {{2, 3, 2}});                   \
    float other = 1.4;                                \
    auto expected = torch::op(GetCpuInput(0), other); \
    auto result = torch::op(GetHpuInput(0), other);   \
    Compare(expected, result);                        \
  }

#define TENSOR_OUT_TEST(op)                                                 \
  TEST_F(HpuOpTest, op##_other_tensor_out) {                                \
    GenerateInputs(2, {{2, 3, 1}, {2, 3, 1}});                              \
    torch::ScalarType dtype = torch::kFloat;                                \
    auto expected = torch::empty((2, 3, 1), dtype);                         \
    auto result =                                                           \
        torch::empty((2, 3, 1), torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), GetCpuInput(1), expected);                    \
    torch::op(GetHpuInput(0), GetHpuInput(1), result);                      \
    Compare(expected, result);                                              \
  }

#define OTHER_SCALAR_OUT_TEST(op)                                           \
  TEST_F(HpuOpTest, op##_other_scalar_out) {                                \
    GenerateInputs(1, {{2, 3, 3}});                                         \
    torch::ScalarType dtype = torch::kFloat;                                \
    float other = 1.1;                                                      \
    auto expected = torch::empty((2, 3, 3), dtype);                         \
    auto result =                                                           \
        torch::empty((2, 3, 3), torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), other, expected);                             \
    torch::op(GetHpuInput(0), other, result);                               \
    Compare(expected, result);                                              \
  }

#define SELF_SCALAR_OUTPLACE_TEST(op)                \
  TEST_F(HpuOpTest, op##_self_scalar) {              \
    GenerateInputs(1, {{2, 2, 1}});                  \
    float self = 2.3;                                \
    auto expected = torch::op(self, GetCpuInput(0)); \
    auto result = torch::op(self, GetHpuInput(0));   \
    Compare(expected, result);                       \
  }

#define SELF_SCALAR_OUT_TEST(op)                                            \
  TEST_F(HpuOpTest, op##_self_scalar_out) {                                 \
    GenerateInputs(1, {{4, 3, 2}});                                         \
    float self = 1;                                                         \
    torch::ScalarType dtype = torch::kFloat;                                \
    auto expected = torch::empty((4, 3, 2), dtype);                         \
    auto result =                                                           \
        torch::empty((4, 3, 2), torch::TensorOptions(dtype).device("hpu")); \
    torch::op(self, GetCpuInput(0), expected);                              \
    torch::op(self, GetHpuInput(0), result);                                \
    Compare(expected, result);                                              \
  }

#define OTHER_SCALAR_INPLACE_TEST(op)        \
  TEST_F(HpuOpTest, op##other_scalar_) {     \
    GenerateInputs(1, {{2, 1}});             \
    float self = 2.3;                        \
    GetCpuInput(0).op(self);                 \
    GetHpuInput(0).op(self);                 \
    Compare(GetCpuInput(0), GetHpuInput(0)); \
  }

#define OTHER_TENSOR_INPLACE_TEST(op)        \
  TEST_F(HpuOpTest, op##other_tensor_) {     \
    GenerateInputs(2, {{2, 1}});             \
    GetCpuInput(0).op(GetCpuInput(1));       \
    GetHpuInput(0).op(GetHpuInput(1));       \
    Compare(GetCpuInput(0), GetHpuInput(0)); \
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