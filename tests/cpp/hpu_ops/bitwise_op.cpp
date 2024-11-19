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

#define TENSOR_TYPE_bool torch::kBool
#define TENSOR_TYPE_int torch::kInt
#define TENSOR_TYPE_uint torch::kUInt8
#define TENSOR_TYPE_short torch::kInt16

#define GET_TENSOR_TYPE(type) TENSOR_TYPE_##type

#define HPU_BITWISE_OUTPLACE_TEST(name, op, dtype)                  \
  TEST_F(HpuOpTest, name) {                                         \
    /* Tensor Tensor inputs */                                      \
    GenerateInputs(2, {{4, 5}, {3, 4, 5}}, GET_TENSOR_TYPE(dtype)); \
    auto expected1 = torch::op(GetCpuInput(0), GetCpuInput(1));     \
    auto result1 = torch::op(GetHpuInput(0), GetHpuInput(1));       \
    Compare(expected1, result1);                                    \
  }

#define HPU_BITWISE_OUTPLACE_SCALAR_TEST(name, op, dtype)   \
  TEST_F(HpuOpTest, name) {                                 \
    GenerateInputs(1, {{3, 4, 5}}, GET_TENSOR_TYPE(dtype)); \
    dtype s = GenerateScalar<dtype>();                      \
    /* Tensor Scalar inputs */                              \
    auto expected1 = torch::op(GetCpuInput(0), s);          \
    auto result1 = torch::op(GetHpuInput(0), s);            \
    Compare(expected1, result1);                            \
    /* Scalar Tensor inputs */                              \
    auto expected2 = torch::op(s, GetCpuInput(0));          \
    auto result2 = torch::op(s, GetHpuInput(0));            \
    Compare(expected2, result2);                            \
  }

#define HPU_BITWISE_INPLACE_TEST(name, op, dtype)                   \
  TEST_F(HpuOpTest, name) {                                         \
    /* Tensor Tensor inputs */                                      \
    GenerateInputs(2, {{3, 4, 5}, {4, 5}}, GET_TENSOR_TYPE(dtype)); \
    GetCpuInput(0).op(GetCpuInput(1));                              \
    GetHpuInput(0).op(GetHpuInput(1));                              \
    Compare(GetCpuInput(0), GetHpuInput(0));                        \
  }

#define HPU_BITWISE_INPLACE_SCALAR_TEST(name, op, dtype) \
  TEST_F(HpuOpTest, name) {                              \
    /* Tensor Scalar inputs */                           \
    GenerateInputs(1, {{4, 5}}, GET_TENSOR_TYPE(dtype)); \
    dtype s = GenerateScalar<dtype>();                   \
    GetCpuInput(0).op(s);                                \
    GetHpuInput(0).op(s);                                \
    Compare(GetCpuInput(0), GetHpuInput(0));             \
  }

#define HPU_BITWISE_OUT_TEST(name, op, dtype_)                          \
  TEST_F(HpuOpTest, name) {                                             \
    /* Tensor Tensor inputs */                                          \
    auto dtype = GET_TENSOR_TYPE(dtype_);                               \
    GenerateInputs(2, {{3, 4, 5}, {4, 5}}, dtype);                      \
    auto exp1 = torch::empty(0, dtype);                                 \
    auto res1 =                                                         \
        torch::empty(0, torch::TensorOptions(dtype).device(c10::kHPU)); \
    torch::op(GetCpuInput(0), GetCpuInput(1), exp1);                    \
    torch::op(GetHpuInput(0), GetHpuInput(1), res1);                    \
    Compare(exp1, res1);                                                \
  }

#define HPU_BITWISE_OUT_SCALAR_TEST(name, op, dtype_)                   \
  TEST_F(HpuOpTest, name) {                                             \
    /* Tensor Scalar inputs */                                          \
    auto dtype = GET_TENSOR_TYPE(dtype_);                               \
    GenerateInputs(1, {{3, 4, 5}}, dtype);                              \
    dtype_ s = GenerateScalar<dtype_>();                                \
    auto exp1 = torch::empty(0, dtype);                                 \
    auto res1 =                                                         \
        torch::empty(0, torch::TensorOptions(dtype).device(c10::kHPU)); \
    torch::op(GetCpuInput(0), s, exp1);                                 \
    torch::op(GetHpuInput(0), s, res1);                                 \
    Compare(exp1, res1);                                                \
  }

class HpuOpTest : public HpuOpTestUtil {};

#define TEST_HPU_BITWISE_OP(op)                                    \
  HPU_BITWISE_OUTPLACE_TEST(op##_tensor, op, uint)                 \
  HPU_BITWISE_OUTPLACE_SCALAR_TEST(op##_scalar, op, bool)          \
  HPU_BITWISE_INPLACE_TEST(op##_inplace_tensor, op##_, short)      \
  HPU_BITWISE_INPLACE_SCALAR_TEST(op##_inplace_scalar, op##_, int) \
  HPU_BITWISE_OUT_TEST(op##_out_tensor, op##_outf, uint)           \
  HPU_BITWISE_OUT_SCALAR_TEST(op##_out_scalar, op##_outf, bool)

TEST_HPU_BITWISE_OP(bitwise_and)
TEST_HPU_BITWISE_OP(bitwise_or)
TEST_HPU_BITWISE_OP(bitwise_xor)

#define HPU_DYNAMIC_BITWISE_INPLACE_TEST(name, op, dtype)        \
  TEST_F(HpuOpDynamicTest, name) {                               \
    {                                                            \
      /* Tensor Tensor inputs */                                 \
      GenerateInputs(2, {{16}, {1}}, GET_TENSOR_TYPE(dtype));    \
      GetCpuInput(0).op(GetCpuInput(1));                         \
      GetHpuInput(0).op(GetHpuInput(1));                         \
      Compare(GetCpuInput(0), GetHpuInput(0));                   \
    }                                                            \
    {                                                            \
      /* Tensor Tensor inputs */                                 \
      GenerateInputs(2, {{128}, {128}}, GET_TENSOR_TYPE(dtype)); \
      GetCpuInput(0).op(GetCpuInput(1));                         \
      GetHpuInput(0).op(GetHpuInput(1));                         \
      Compare(GetCpuInput(0), GetHpuInput(0));                   \
    }                                                            \
  }

class HpuOpDynamicTest : public HpuOpTestUtil {};

#define TEST_HPU_DYNAMIC_BITWISE_OP(op) \
  HPU_DYNAMIC_BITWISE_INPLACE_TEST(op##_inplace_tensor, op##_, bool)

TEST_HPU_DYNAMIC_BITWISE_OP(bitwise_or)

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