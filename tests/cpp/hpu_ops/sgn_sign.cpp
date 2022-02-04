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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, sgn) {
  GenerateInputs(1);
  auto exp = torch::sgn(GetCpuInput(0));
  auto res = torch::sgn(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sgn_bf16) {
  GenerateInputs(1, {{5, 6}}, {torch::kBFloat16});
  auto exp = torch::sgn(GetCpuInput(0));
  auto res = torch::sgn(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sgn_inplace) {
  GenerateInputs(1, torch::kFloat);
  auto expected = GetCpuInput(0);
  auto result = GetHpuInput(0);

  expected = torch::sgn(expected);
  result = torch::sgn(result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sgn_out) {
  GenerateInputs(1, torch::kFloat);
  auto expected = torch::empty(0);
  auto result = torch::empty(0, torch::TensorOptions().device("hpu"));

  torch::sgn_outf(GetCpuInput(0), expected);
  torch::sgn_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sign) {
  GenerateInputs(1);
  auto exp = torch::sign(GetCpuInput(0));
  auto res = torch::sign(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sign_bf16) {
  GenerateInputs(1, {{5, 6}}, {torch::kBFloat16});
  auto exp = torch::sign(GetCpuInput(0));
  auto res = torch::sign(GetHpuInput(0));
  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, sign_inplace) {
  GenerateInputs(1, torch::kFloat);
  auto expected = GetCpuInput(0);
  auto result = GetHpuInput(0);

  expected = torch::sign(expected);
  result = torch::sign(result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, sign_out) {
  GenerateInputs(1, torch::kFloat);
  auto expected = torch::empty(0);
  auto result = torch::empty(0, torch::TensorOptions().device("hpu"));

  torch::sign_outf(GetCpuInput(0), expected);
  torch::sign_outf(GetHpuInput(0), result);

  Compare(expected, result);
}
