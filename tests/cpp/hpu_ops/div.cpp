/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class DivScalarHpuOpTest : public HpuOpTestUtil {};

TEST_F(DivScalarHpuOpTest, div_scalar_int_int) {
  GenerateInputs(1, {{4, 5}}, torch::kInt);
  auto expected = torch::div(GetCpuInput(0), 11);
  auto result = torch::div(GetHpuInput(0), 11);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_float_int) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  auto expected = torch::div(GetCpuInput(0), 11);
  auto result = torch::div(GetHpuInput(0), 11);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_int) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  auto expected = torch::div(GetCpuInput(0), 11);
  auto result = torch::div(GetHpuInput(0), 11);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_int_float) {
  GenerateInputs(1, {{4, 5}}, torch::kInt);
  auto expected = torch::div(GetCpuInput(0), 1.1);
  auto result = torch::div(GetHpuInput(0), 1.1);
  Compare(expected, result);
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_float) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  auto expected = torch::div(GetCpuInput(0), 1.1);
  auto result = torch::div(GetHpuInput(0), 1.1);
  Compare(expected, result, 2e-2, 2e-2);
}

TEST_F(DivScalarHpuOpTest, div_scalar_float_float) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  auto expected = torch::div(GetCpuInput(0), 1.1);
  auto result = torch::div(GetHpuInput(0), 1.1);
  Compare(expected, result);
}
// inplace scalar ops
TEST_F(DivScalarHpuOpTest, div_scalar_float_int_) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  GetCpuInput(0).div_(11);
  GetHpuInput(0).div_(11);
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_int_) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  GetCpuInput(0).div_(11);
  GetHpuInput(0).div_(11);
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(DivScalarHpuOpTest, div_scalar_bfloat_float_) {
  GenerateInputs(1, {{4, 5}}, torch::kBFloat16);
  GetCpuInput(0).div_(1.1);
  GetHpuInput(0).div_(1.1);
  Compare(GetCpuInput(0), GetHpuInput(0), 2e-2, 2e-2);
}

TEST_F(DivScalarHpuOpTest, div_scalar_float_float_) {
  GenerateInputs(1, {{4, 5}}, torch::kFloat);
  GetCpuInput(0).div_(1.1);
  GetHpuInput(0).div_(1.1);
  Compare(GetCpuInput(0), GetHpuInput(0));
}