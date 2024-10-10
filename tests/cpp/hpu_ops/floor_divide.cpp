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
#define SIZE(...) __VA_ARGS__

#define HPU_FLOOR_DIVIDE_INPLACE_TEST(                       \
    name, in_size1, in_size2, dtype1, dtype2)                \
  TEST_F(HpuOpTest, name) {                                  \
    auto self = torch::randn({in_size1}).to(dtype1);         \
    auto other = torch::randint(1, 100, {in_size2}, dtype2); \
    auto self_h = self.to(torch::kHPU);                      \
    auto other_h = other.to(torch::kHPU);                    \
    self.floor_divide_(other);                               \
    self_h.floor_divide_(other_h);                           \
    Compare(self, self_h);                                   \
  }

#define HPU_FLOOR_DIVIDE_OUT_TEST(name, in_size1, in_size2, dtype1, dtype2)    \
  TEST_F(HpuOpTest, name) {                                                    \
    torch::manual_seed(0);                                                     \
    auto self = torch::randn({in_size1}).to(dtype1);                           \
    auto other = torch::randint(1, 100, {in_size2}, dtype2);                   \
    auto self_h = self.to(torch::kHPU);                                        \
    auto other_h = other.to(torch::kHPU);                                      \
    auto expected = torch::empty(0, dtype1);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype1).device("hpu")); \
    torch::floor_divide_outf(self, other, expected);                           \
    torch::floor_divide_outf(self_h, other_h, result);                         \
    Compare(expected, result);                                                 \
  }
#define HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(                       \
    op, in_size1, in_size2, datatype_1, datatype_2)                      \
  TEST_F(HpuOpTest, op) {                                                \
    torch::ScalarType dtype_1 = datatype_1;                              \
    torch::ScalarType dtype_2 = datatype_2;                              \
    GenerateInputs(2, {in_size1, in_size2}, {dtype_1, dtype_2});         \
    auto expected = torch::floor_divide(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::floor_divide(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                           \
  }

#define HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(op, in_size, datatype, scalartype) \
  TEST_F(HpuOpTest, op) {                                                     \
    torch::ScalarType dtype = datatype;                                       \
    auto other = GenerateScalar<scalartype>();                                \
    GenerateInputs(1, {in_size}, {dtype});                                    \
    auto expected = torch::floor_divide(GetCpuInput(0), other);               \
    auto result = torch::floor_divide(GetHpuInput(0), other);                 \
    Compare(expected, result);                                                \
  }

#define HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(  \
    op, in_size, datatype, scalartype)         \
  TEST_F(HpuOpTest, op) {                      \
    torch::ScalarType dtype = datatype;        \
    auto other = GenerateScalar<scalartype>(); \
    GenerateInputs(1, {in_size}, {dtype});     \
    GetCpuInput(0).floor_divide_(other);       \
    GetHpuInput(0).floor_divide_(other);       \
    Compare(GetCpuInput(0), GetHpuInput(0));   \
  }

class HpuOpTest : public HpuOpTestUtil {};

// aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other)
HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_float_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_float_float_broadcast,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 1, 1}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_bfloat_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kFloat)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_bfloat_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kInt)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_float_bfloat,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_float_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kInt)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_int_int,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kInt,
    torch::kInt)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_float_byte,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kFloat,
    torch::kByte)

HPU_FLOOR_DIVIDE_INPLACE_TEST(
    floor_divide_inplace_char_char,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kChar,
    torch::kChar)

// aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out)
HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_bfloat,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kInt)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_bfloat_bfloat,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 1, 1}),
    torch::kBFloat16,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_bfloat_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kInt)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_bfloat_float,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kBFloat16,
    torch::kFloat)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_int_int,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kInt,
    torch::kInt)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_float_byte,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kFloat,
    torch::kByte)

HPU_FLOOR_DIVIDE_OUT_TEST(
    floor_divide_out_char_char,
    SIZE({8, 4, 4, 4}),
    SIZE({8, 4, 4, 4}),
    torch::kChar,
    torch::kChar)

// aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other)
HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_bfloat_float,
    SIZE({8, 4, 8, 4, 4}),
    torch::kBFloat16,
    float)

HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_int_int,
    SIZE({3, 6, 7}),
    torch::kInt,
    int)

HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_float_int,
    SIZE({8, 4, 4, 4}),
    torch::kFloat,
    int)

HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_float_double,
    SIZE({3, 4}),
    torch::kFloat,
    double)

HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_float_float,
    SIZE({1024}),
    torch::kFloat,
    float)

HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_bfloat_int,
    SIZE({8, 4, 1, 4}),
    torch::kBFloat16,
    int)

HPU_FLOOR_DIVIDE_INPLACE_SCALAR_TEST(
    floor_divide_inplace_scalar_char_int,
    SIZE({8, 4, 1, 4}),
    torch::kChar,
    int)

// aten::floor_divide(Tensor self, Tensor other)
HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_float,
    SIZE({8, 24, 24, 3}),
    SIZE({1, 24, 24, 1}),
    torch::kFloat,
    torch::kFloat)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_float_bfloat,
    SIZE({8, 24, 24, 5}),
    SIZE({5}),
    torch::kFloat,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    DISABLED_floor_divide_usual_bfloat_bfloat,
    SIZE({8, 4, 5}),
    SIZE({4, 5}),
    torch::kBFloat16,
    torch::kBFloat16)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_bfloat_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 1, 1}),
    torch::kBFloat16,
    torch::kInt)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_float_int,
    SIZE({8, 4, 8, 4}),
    SIZE({8, 4, 8, 4}),
    torch::kFloat,
    torch::kInt)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_int_int,
    SIZE({2, 4}),
    SIZE({4}),
    torch::kInt,
    torch::kInt)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_byte_byte,
    SIZE({2, 4}),
    SIZE({2, 4}),
    torch::kByte,
    torch::kByte)

HPU_FLOOR_DIVIDE_USUAL_TENSOR_TENSOR_TEST(
    floor_divide_usual_char_char,
    SIZE({2, 4}),
    SIZE({2, 4}),
    torch::kChar,
    torch::kChar)

// aten::floor_divide.Scalar(Tensor self, Scalar other)
HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_int_int,
    SIZE({3, 6, 7}),
    torch::kInt,
    int)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_float_int,
    SIZE({8, 4, 4, 4}),
    torch::kFloat,
    int)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_float_double,
    SIZE({3, 4}),
    torch::kFloat,
    double)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_float_float,
    SIZE({1024}),
    torch::kFloat,
    float)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_bfloat_int,
    SIZE({8, 4, 1, 4}),
    torch::kBFloat16,
    int)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_int_float,
    SIZE({8, 4, 1, 4}),
    torch::kInt,
    float)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_byte_int,
    SIZE({8, 4, 1, 4}),
    torch::kByte,
    int)

HPU_FLOOR_DIVIDE_USUAL_SCALAR_TEST(
    floor_divide_usual_scalar_char_float,
    SIZE({8, 4, 1, 4}),
    torch::kChar,
    float)
