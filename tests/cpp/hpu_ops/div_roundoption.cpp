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

// Limits for generated values of variants of Int
#define MIN_INT_VALUE_GENERATED -360
#define MAX_INT_VALUE_GENERATED -1

#define MIN_INT8_VALUE_GENERATED 1
#define MAX_INT8_VALUE_GENERATED 50

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, divroundTrueDouble) {
  GenerateInputs(2, torch::kDouble);
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(GetCpuInput(0), GetCpuInput(1), mode);
  auto result = torch::div(GetHpuInput(0), GetCpuInput(1), mode);

  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundTrueBFloat16) {
  GenerateInputs(2, torch::kBFloat16);
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(GetCpuInput(0), GetCpuInput(1), mode);
  auto result = torch::div(GetHpuInput(0), GetCpuInput(1), mode);
  // TPC Kernel's precision, slightly differs fro CPU version for bfloat16
  // Hence increased tolerance
  Compare(expected, result, 0.1, 0.1);
}

TEST_F(HpuOpTest, divroundFloor) {
  GenerateInputs(2);
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::div(GetCpuInput(0), GetCpuInput(1), "floor");
  auto result = torch::div(GetHpuInput(0), GetCpuInput(1), "floor");

  Compare(expected, result);
}

// TODO: Use Test Util class (HpuOpTestUtil) (for each Int type test ) after
// following issue fixed: [PO-165] Random Generation Fails for Byte, Char,
// Short, Int, Long with Test Util class (HpuOpTestUtil)
TEST_F(HpuOpTest, divroundTrueBroadcast) {
  auto A = torch::randn({4, 3});
  auto B = torch::randn({1, 3});
  auto hA = A.to("hpu");
  auto hB = B.to("hpu");
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(A, B, mode);
  auto result = torch::div(hA, hB, mode);
  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundTrueTypePromoIntFloat) {
  const std::vector<int64_t> tensor_size = {2, 5};
  auto A = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto B = torch::randn(tensor_size);
  auto hA = A.to("hpu");
  auto hB = B.to("hpu");
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(A, B, mode);
  auto result = torch::div(hA, hB, mode);
  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundTruncateTypePromoInt8Int8) {
  const std::vector<int64_t> tensor_size = {4, 6};
  auto A = torch::randint(
      MIN_INT8_VALUE_GENERATED,
      MAX_INT8_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt8));
  auto B = torch::randint(
      MIN_INT8_VALUE_GENERATED,
      MAX_INT8_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt8));
  auto hA = A.to("hpu");
  auto hB = B.to("hpu");
  auto expected = torch::div(A, B, "trunc");
  auto result = torch::div(hA, hB, "trunc");
  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundTruncateTypePromoIntInt) {
  const std::vector<int64_t> tensor_size = {4, 6, 5, 3};
  auto A = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto B = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto hA = A.to("hpu");
  auto hB = B.to("hpu");
  auto expected = torch::div(A, B, "trunc");
  auto result = torch::div(hA, hB, "trunc");
  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundFloorTypePromoIntInt) {
  const std::vector<int64_t> tensor_size = {3, 2, 6, 4};
  auto A = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto B = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto hA = A.to("hpu");
  auto hB = B.to("hpu");
  auto expected = torch::div(A, B, "floor");
  auto result = torch::div(hA, hB, "floor");
  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundNoneTypePromoIntInt) {
  const std::vector<int64_t> tensor_size = {2, 3, 2, 4};
  auto A = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto B = torch::randint(
      MIN_INT_VALUE_GENERATED,
      MAX_INT_VALUE_GENERATED,
      tensor_size,
      torch::dtype(torch::kInt));
  auto hA = A.to("hpu");
  auto hB = B.to("hpu");
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(A, B, mode);
  auto result = torch::div(hA, hB, mode);
  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundTrueTypePromoIntInt1D) {
  std::vector<int> tensorDataA = {29, 73, -37, -317, 99, 81, -98, -72};
  std::vector<int> tensorDataB = {-10, 3, -7, 17, -11, 3, -7, 6};
  auto A = torch::from_blob(
      tensorDataA.data(), tensorDataA.size(), dtype(torch::kInt));
  auto B = torch::from_blob(
      tensorDataB.data(), tensorDataB.size(), dtype(torch::kInt));

  auto hA = A.to("hpu");
  auto hB = B.to("hpu");

  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(A, B, mode);
  auto result = torch::div(hA, hB, mode);

  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundTruncTypePromoIntInt1D) {
  std::vector<int> tensorDataA = {29, 73, -37, -317, 99, 81, -98, -72};
  std::vector<int> tensorDataB = {-10, 3, -7, 17, -11, 3, -7, 6};
  auto A = torch::from_blob(
      tensorDataA.data(), tensorDataA.size(), dtype(torch::kInt));
  auto B = torch::from_blob(
      tensorDataB.data(), tensorDataB.size(), dtype(torch::kInt));

  auto hA = A.to("hpu");
  auto hB = B.to("hpu");

  auto expected = torch::div(A, B, "trunc");
  auto result = torch::div(hA, hB, "trunc");

  Compare(expected, result);
}

TEST_F(HpuOpTest, divroundFloorTypePromoIntInt1D) {
  std::vector<int> tensorDataA = {29, 73, -37, -317, 99, 81, -98, -72};
  std::vector<int> tensorDataB = {-10, 3, -7, 17, -11, 3, -7, 6};
  auto A = torch::from_blob(
      tensorDataA.data(), tensorDataA.size(), dtype(torch::kInt));
  auto B = torch::from_blob(
      tensorDataB.data(), tensorDataB.size(), dtype(torch::kInt));

  auto hA = A.to("hpu");
  auto hB = B.to("hpu");

  auto expected = torch::div(A, B, "floor");
  auto result = torch::div(hA, hB, "floor");

  Compare(expected, result);
}

TEST_F(HpuOpTest, div_inplace_f32) {
  GenerateInputs(1, torch::kFloat);
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto other = GenerateScalar<float>();

  GetCpuInput(0).div_(other, mode);
  GetHpuInput(0).div_(other, mode);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, div_inplace_f32int) {
  GenerateInputs(1, torch::kFloat);
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto other = GenerateScalar<int>();

  GetCpuInput(0).div_(other, mode);
  GetHpuInput(0).div_(other, mode);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, div_inplace_bf16int8) {
  GenerateInputs(2, {torch::kBFloat16, torch::kInt8});
  c10::optional<c10::string_view> mode = c10::nullopt;

  GetCpuInput(0).div_(GetCpuInput(1), mode);
  GetHpuInput(0).div_(GetCpuInput(1), mode);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, div_scalar_int) {
  GenerateInputs(1, torch::kInt);
  auto other = GenerateScalar<int>();

  auto expected = torch::div(GetCpuInput(0), other, /* rounding_mode*/ "trunc");
  auto result = torch::div(GetHpuInput(0), other, /* rounding_mode*/ "trunc");

  Compare(expected, result);
}

TEST_F(HpuOpTest, div_scalar_int8f32) {
  GenerateInputs(1, torch::kInt8);
  auto other = GenerateScalar<float>();

  auto expected = torch::div(GetCpuInput(0), other, /* rounding_mode*/ "floor");
  auto result = torch::div(GetHpuInput(0), other, /* rounding_mode*/ "floor");

  Compare(expected, result);
}

TEST_F(HpuOpTest, div_scalar_bf16int) {
  GenerateInputs(1, torch::kBFloat16);
  auto other = GenerateScalar<int>();
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(GetCpuInput(0), other, mode);
  auto result = torch::div(GetHpuInput(0), other, mode);

  Compare(expected, result);
}

TEST_F(HpuOpTest, div_scalar_bf16f32) {
  GenerateInputs(1, torch::kBFloat16);
  auto other = GenerateScalar<float>();
  c10::optional<c10::string_view> mode = c10::nullopt;
  auto expected = torch::div(GetCpuInput(0), other, mode);
  auto result = torch::div(GetHpuInput(0), other, mode);

  Compare(expected, result);
}

TEST_F(HpuOpTest, div_out_f32) {
  GenerateInputs(2);
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::div_outf(
      GetCpuInput(0), GetCpuInput(1), /* rounding_mode*/ "trunc", expected);
  torch::div_outf(
      GetHpuInput(0), GetHpuInput(1), /* rounding_mode*/ "trunc", result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, div_out_f32int8) {
  GenerateInputs(2, {torch::kFloat, torch::kInt8});
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::div_outf(
      GetCpuInput(0), GetCpuInput(1), /* rounding_mode*/ "floor", expected);
  torch::div_outf(
      GetHpuInput(0), GetHpuInput(1), /* rounding_mode*/ "floor", result);

  Compare(expected, result);
}
