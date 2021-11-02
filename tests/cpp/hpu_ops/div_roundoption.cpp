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

// Limits for generated values of variants of Int
#define MIN_INT_VALUE_GENERATED -360
#define MAX_INT_VALUE_GENERATED 360

#define MIN_INT8_VALUE_GENERATED -50
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
