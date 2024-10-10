/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "util.h"

#define SIZE(...) __VA_ARGS__

/*shift-size is greater than the number of bits a dtype supports
its undefined behavior and not portable across all hardwares
so using other modulo */

#define HPU_LEFT_SHIFT_OUT_TEST(                                               \
    name, in_size1, in_size2, dtype1, dtype2, dtype_size)                      \
  TEST_F(HpuOpTest, name) {                                                    \
    GenerateInputs(2, {in_size1, in_size2}, {dtype1, dtype2});                 \
    GetCpuInput(1) = GetCpuInput(1) % dtype_size;                              \
    auto other = GetCpuInput(1).to(torch::kHPU);                               \
    auto expected = torch::empty(0, dtype1);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype1).device("hpu")); \
    torch::bitwise_left_shift_outf(GetCpuInput(0), GetCpuInput(1), expected);  \
    torch::bitwise_left_shift_outf(GetHpuInput(0), GetHpuInput(1), result);    \
    Compare(expected, result);                                                 \
  }

#define HPU_LEFT_SHIFT_USUAL_TEST(                                             \
    name, in_size1, in_size2, dtype1, dtype2, dtype_size)                      \
  TEST_F(HpuOpTest, name) {                                                    \
    GenerateInputs(2, {in_size1, in_size2}, {dtype1, dtype2});                 \
    GetCpuInput(1) = GetCpuInput(1) % dtype_size;                              \
    auto other = GetCpuInput(1).to(torch::kHPU);                               \
    auto expected = torch::bitwise_left_shift(GetCpuInput(0), GetCpuInput(1)); \
    auto result = torch::bitwise_left_shift(GetHpuInput(0), GetHpuInput(1));   \
    Compare(expected, result);                                                 \
  }

#define HPU_LEFT_SHIFT_INPLACE_TEST(                           \
    name, op, in_size1, in_size2, dtype1, dtype2, dtype_size)  \
  TEST_F(HpuOpTest, name) {                                    \
    GenerateInputs(2, {in_size1, in_size2}, {dtype1, dtype2}); \
    GetCpuInput(1) = GetCpuInput(1) % dtype_size;              \
    auto other = GetCpuInput(1).to(torch::kHPU);               \
    GetCpuInput(0).op(GetCpuInput(1));                         \
    GetHpuInput(0).op(GetHpuInput(1));                         \
    Compare(GetCpuInput(0), GetHpuInput(0));                   \
  }

#define HPU_RIGHT_SHIFT_OUT_TEST(                                              \
    name, in_size1, in_size2, dtype1, dtype2, dtype_size)                      \
  TEST_F(HpuOpTest, name) {                                                    \
    GenerateInputs(2, {in_size1, in_size2}, {dtype1, dtype2});                 \
    GetCpuInput(1) = GetCpuInput(1) % dtype_size;                              \
    auto other = GetCpuInput(1).to(torch::kHPU);                               \
    auto expected = torch::empty(0, dtype1);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype1).device("hpu")); \
    torch::bitwise_right_shift_outf(GetCpuInput(0), GetCpuInput(1), expected); \
    torch::bitwise_right_shift_outf(GetHpuInput(0), other, result);            \
    Compare(expected, result);                                                 \
  }

#define HPU_RIGHT_SHIFT_USUAL_TEST(                                  \
    name, in_size1, in_size2, dtype1, dtype2, dtype_size)            \
  TEST_F(HpuOpTest, name) {                                          \
    GenerateInputs(2, {in_size1, in_size2}, {dtype1, dtype2});       \
    GetCpuInput(1) = GetCpuInput(1) % dtype_size;                    \
    auto other = GetCpuInput(1).to(torch::kHPU);                     \
    auto expected =                                                  \
        torch::bitwise_right_shift(GetCpuInput(0), GetCpuInput(1));  \
    auto result = torch::bitwise_right_shift(GetHpuInput(0), other); \
    Compare(expected, result);                                       \
  }

#define HPU_RIGHT_SHIFT_INPLACE_TEST(                          \
    name, op, in_size1, in_size2, dtype1, dtype2, dtype_size)  \
  TEST_F(HpuOpTest, name) {                                    \
    GenerateInputs(2, {in_size1, in_size2}, {dtype1, dtype2}); \
    GetCpuInput(1) = GetCpuInput(1) % dtype_size;              \
    auto other = GetCpuInput(1).to(torch::kHPU);               \
    GetCpuInput(0).op(GetCpuInput(1));                         \
    GetHpuInput(0).op(other);                                  \
    Compare(GetCpuInput(0), GetHpuInput(0));                   \
  }

#define HPU_SCALAR_INPLACE_TEST(name, op, dtype, dtype_size) \
  TEST_F(HpuOpTest, name) {                                  \
    GenerateInputs(1, {dtype});                              \
    int other = GenerateScalar<int>(1, dtype_size);          \
    GetCpuInput(0).op(other);                                \
    GetHpuInput(0).op(other);                                \
    Compare(GetCpuInput(0), GetHpuInput(0));                 \
  }

#define HPU_SCALAR_OUT_TEST(name, op, dtype, dtype_size)                      \
  TEST_F(HpuOpTest, name) {                                                   \
    GenerateInputs(1, {dtype});                                               \
    int other = GenerateScalar<int>(1, dtype_size);                           \
    auto expected = torch::empty(0, dtype);                                   \
    auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu")); \
    torch::op(GetCpuInput(0), other, expected);                               \
    torch::op(GetHpuInput(0), other, result);                                 \
    Compare(expected, result);                                                \
  }

#define HPU_SCALAR_USUAL_TEST(name, op, dtype, dtype_size) \
  TEST_F(HpuOpTest, name) {                                \
    GenerateInputs(1, {dtype});                            \
    int other = GenerateScalar<int>(1, dtype_size);        \
    auto expected = torch::op(GetCpuInput(0), other);      \
    auto result = torch::op(GetHpuInput(0), other);        \
    Compare(expected, result);                             \
  }

#define BITSHIFT_SCALAR_TEST(name, op)                                   \
  HPU_SCALAR_USUAL_TEST(name##_scal, op, torch::kInt16, 16)              \
  HPU_SCALAR_OUT_TEST(name##_scal_out_i32, op##_outf, torch::kInt32, 32) \
  HPU_SCALAR_INPLACE_TEST(name##_scal_inplace, op##_, torch::kInt8, 8)

class HpuOpTest : public HpuOpTestUtil {};

HPU_LEFT_SHIFT_USUAL_TEST(
    left_shift_i32,
    SIZE({8, 24, 24, 3}),
    SIZE({8, 24, 24, 3}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_LEFT_SHIFT_USUAL_TEST(
    left_shift_bc_i32,
    SIZE({2, 1, 6}),
    SIZE({2, 8, 6}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_LEFT_SHIFT_USUAL_TEST(
    left_shift_i8,
    SIZE({2, 2}),
    SIZE({2, 2}),
    torch::kInt8,
    torch::kInt8,
    8)

HPU_LEFT_SHIFT_OUT_TEST(
    left_shift_out_i32,
    SIZE({2, 3, 4}),
    SIZE({2, 3, 4}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_LEFT_SHIFT_INPLACE_TEST(
    left_shift_,
    bitwise_left_shift_,
    SIZE({2, 3, 4}),
    SIZE({2, 3, 4}),
    torch::kUInt8,
    torch::kUInt8,
    8)

// BITSHIFT_SCALAR_TEST(left_shiftt, bitwise_left_shift)

HPU_LEFT_SHIFT_INPLACE_TEST(
    ilshift_tensor_bc,
    __ilshift__,
    SIZE({8, 12, 12}),
    SIZE({1, 12, 12}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_SCALAR_INPLACE_TEST(lshift_scalar_u8_, __lshift__, torch::kUInt8, 8)

HPU_SCALAR_INPLACE_TEST(ilshift_scalar_i16_, __ilshift__, torch::kInt16, 16)

HPU_SCALAR_INPLACE_TEST(left_shift_scal_, bitwise_left_shift_, torch::kUInt8, 8)

HPU_SCALAR_INPLACE_TEST(
    left_shift_scal_i32_,
    bitwise_left_shift_,
    torch::kInt32,
    32)

HPU_SCALAR_INPLACE_TEST(rshift_scalar_i8_, __rshift__, torch::kInt8, 8)

HPU_SCALAR_INPLACE_TEST(irshift_scalar_i16_, __irshift__, torch::kInt16, 16)

HPU_RIGHT_SHIFT_USUAL_TEST(
    right_shift_mod_i32,
    SIZE({8, 24, 24, 3}),
    SIZE({8, 24, 1, 3}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_RIGHT_SHIFT_USUAL_TEST(
    right_shift_i16,
    SIZE({2, 2}),
    SIZE({2, 2}),
    torch::kInt16,
    torch::kInt16,
    16)

HPU_RIGHT_SHIFT_USUAL_TEST(
    right_shift_i16_bc,
    SIZE({1, 2}),
    SIZE({2, 2}),
    torch::kInt16,
    torch::kInt16,
    16)

HPU_RIGHT_SHIFT_USUAL_TEST(
    right_shift_bc_i16,
    SIZE({2, 3, 4}),
    SIZE({1, 3, 4}),
    torch::kInt16,
    torch::kInt16,
    16)

HPU_RIGHT_SHIFT_INPLACE_TEST(
    right_shift_i32,
    bitwise_right_shift_,
    SIZE({2, 4, 4, 8}),
    SIZE({2, 4, 4, 8}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_LEFT_SHIFT_OUT_TEST(
    right_shift_out_tensor_i32,
    SIZE({2, 3, 4}),
    SIZE({2, 3, 4}),
    torch::kInt32,
    torch::kInt32,
    32)

// BITSHIFT_SCALAR_TEST(right_shift, bitwise_right_shift)

HPU_RIGHT_SHIFT_INPLACE_TEST(
    irshift_tensor_bc,
    __irshift__,
    SIZE({8, 28, 28}),
    SIZE({1, 28, 28}),
    torch::kInt32,
    torch::kInt32,
    32)

HPU_RIGHT_SHIFT_INPLACE_TEST(
    rshift_tensor,
    __rshift__,
    SIZE({4, 1, 3}),
    SIZE({4, 1, 3}),
    torch::kUInt8,
    torch::kUInt8,
    8)

// self is scalar, and cannot use the macro above
// TODO (SW-137838): fix or remove this test case
TEST_F(HpuOpTest, DISABLED_left_shift_scal_ten) {
  GenerateInputs(1, {torch::kInt32});
  int self = 100;
  auto expected = torch::bitwise_left_shift(self, GetCpuInput(0));
  auto result = torch::bitwise_left_shift(self, GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, right_shift_scal_ten_i32) {
  int self = GenerateScalar<int>(44, 50);
  auto t2 = torch::tensor({10, 20});
  auto tensor2 = torch::tensor({10, 20}, "hpu");
  auto expected = torch::bitwise_right_shift(self, t2);
  auto result = torch::bitwise_right_shift(self, tensor2);
  Compare(expected, result);
}

// lshift and rshift Out variants are not supported in CPU
// so used usual variants for reference

TEST_F(HpuOpTest, ls_out) {
  auto dtype = torch::kInt32;
  GenerateInputs(1, {dtype});
  int other = GenerateScalar<int>(1, 32);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = GetCpuInput(0).__lshift__(other);
  torch::__lshift___outf(GetHpuInput(0), other, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, lt_out) {
  auto dtype = torch::kInt32;
  GenerateInputs(2, {dtype, torch::kUInt8});
  GetCpuInput(1) = GetCpuInput(1) % 8;
  auto other = GetCpuInput(1).to(torch::kHPU);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = GetCpuInput(0).__lshift__(GetCpuInput(1));
  torch::__lshift___outf(GetHpuInput(0), other, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, rt_out) {
  auto dtype = torch::kInt16;
  GenerateInputs(2, {dtype, torch::kInt8});
  GetCpuInput(1) = GetCpuInput(1) % 16;
  auto other = GetCpuInput(1).to(torch::kHPU);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = GetCpuInput(0).__rshift__(GetCpuInput(1));
  torch::__rshift___outf(GetHpuInput(0), other, res);

  Compare(exp, res);
}

TEST_F(HpuOpTest, rs_out) {
  auto dtype = torch::kInt8;
  GenerateInputs(1, {dtype});
  int other = GenerateScalar<int>(1, 8);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  auto exp = GetCpuInput(0).__rshift__(other);
  torch::__rshift___outf(GetHpuInput(0), other, res);

  Compare(exp, res);
}
