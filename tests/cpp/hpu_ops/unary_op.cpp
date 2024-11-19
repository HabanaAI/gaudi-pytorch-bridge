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
#define SIZE(...) __VA_ARGS__

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

// "aten::is_floating_point(Tensor self) -> bool"
#define HPU_UNARY_USUAL_BOOL_TEST(name, op, dtype, in_size) \
  TEST_F(HpuOpTest, name) {                                 \
    GenerateInputs(1, {in_size}, {dtype});                  \
    auto expected = torch::op(GetCpuInput(0));              \
    auto result = torch::op(GetHpuInput(0));                \
    EXPECT_EQ(expected, result);                            \
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
HPU_UNARY_USUAL_TEST(trunc)
HPU_UNARY_USUAL_TEST(expm1)

HPU_UNARY_INPLACE_TEST(arccos_)
HPU_UNARY_INPLACE_TEST(arccosh_)
HPU_UNARY_INPLACE_TEST(arcsin_)
HPU_UNARY_INPLACE_TEST(arcsinh_)
HPU_UNARY_INPLACE_TEST(arctan_)
HPU_UNARY_INPLACE_TEST(arctanh_)
HPU_UNARY_INPLACE_TEST(sinh_)
HPU_UNARY_INPLACE_TEST(tan_)
HPU_UNARY_INPLACE_TEST(trunc_)
HPU_UNARY_INPLACE_TEST(expm1_)

HPU_UNARY_OUT_TEST(arccos_outf)
HPU_UNARY_OUT_TEST(arccosh_outf)
HPU_UNARY_OUT_TEST(arcsin_outf)
HPU_UNARY_OUT_TEST(arcsinh_outf)
HPU_UNARY_OUT_TEST(arctan_outf)
HPU_UNARY_OUT_TEST(arctanh_outf)
HPU_UNARY_OUT_TEST(sinh_outf)
HPU_UNARY_OUT_TEST(tan_outf)
HPU_UNARY_OUT_TEST(trunc_outf)
HPU_UNARY_OUT_TEST(expm1_outf)

HPU_UNARY_USUAL_BOOL_TEST(is_neg_float, is_neg, SIZE({4, 5, 6}), torch::kFloat)
HPU_UNARY_USUAL_BOOL_TEST(is_neg_bf16, is_neg, SIZE({5, 6}), torch::kBFloat16)

// is_nonzero expect only size of 1
HPU_UNARY_USUAL_BOOL_TEST(is_nonzero, is_nonzero, SIZE({}), torch::kFloat)
HPU_UNARY_USUAL_BOOL_TEST(is_nonzero_bool, is_nonzero, SIZE({1}), torch::kBool)
HPU_UNARY_USUAL_BOOL_TEST(
    is_nonzero_bf16,
    is_nonzero,
    SIZE({}),
    torch::kBFloat16)

HPU_UNARY_USUAL_BOOL_TEST(
    is_float,
    is_floating_point,
    SIZE({4, 5, 6}),
    torch::kFloat)
HPU_UNARY_USUAL_BOOL_TEST(
    is_float_bf16,
    is_floating_point,
    SIZE({5, 6}),
    torch::kBFloat16)
