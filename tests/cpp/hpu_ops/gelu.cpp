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

TEST_F(HpuOpTest, gelu_Float) {
  GenerateInputs(1);

  auto expected = torch::gelu(GetCpuInput(0));
  auto result = torch::gelu(GetHpuInput(0));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, gelu_BFloat16) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::gelu(GetCpuInput(0));
  auto result = torch::gelu(GetHpuInput(0));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, gelu_Float_appx) {
  GenerateInputs(1);
  std::string apprx = "none";
  auto expected = torch::gelu(GetCpuInput(0), apprx);
  auto result = torch::gelu(GetHpuInput(0), apprx);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, gelu_BFloat16_appx) {
  GenerateInputs(1, torch::kBFloat16);
  std::string apprx = "tanh";
  auto expected = torch::gelu(GetCpuInput(0), apprx);
  auto result = torch::gelu(GetHpuInput(0), apprx);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, gelu_out_float) {
  GenerateInputs(1, {{4, 5, 6}});
  torch::ScalarType dtype = torch::kFloat32;

  auto expected = torch::empty(0, dtype);

  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::gelu_outf(GetCpuInput(0), "none", expected);
  torch::gelu_outf(GetHpuInput(0), "none", result);

  Compare(expected, result);
}

/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_out_bfloat) {
  GenerateInputs(1, {{4, 5, 6}}, torch::kBFloat16);
  torch::ScalarType dtype = torch::kBFloat16;

  auto expected = torch::empty(0, dtype);

  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::gelu_outf(GetCpuInput(0), "tanh", expected);
  torch::gelu_outf(GetHpuInput(0), "tanh", result);

  Compare(expected, result, 1e-2, 1e-2);
}
/* FAILING CASES */
/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_backwardFloat) {
  GenerateInputs(2);

  auto expected = torch::gelu_backward(GetCpuInput(1), GetCpuInput(0));
  auto result = torch::gelu_backward(GetHpuInput(1), GetHpuInput(0));

  Compare(expected, result, 2e-2, 2e-2);
}

/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_backwardBFloat16) {
  GenerateInputs(2, torch::kBFloat16);

  auto expected = torch::gelu_backward(GetCpuInput(1), GetCpuInput(0));
  auto result = torch::gelu_backward(GetHpuInput(1), GetHpuInput(0));

  Compare(expected, result, 2e-2, 2e-2);
}

/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_backwardFloat_approx) {
  GenerateInputs(2);
  std::string apprx = "none";
  auto expected = torch::gelu_backward(GetCpuInput(1), GetCpuInput(0), apprx);
  auto result = torch::gelu_backward(GetHpuInput(1), GetHpuInput(0), apprx);

  Compare(expected, result, 2e-2, 2e-2);
}

/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_backwardBFloat16_approx) {
  GenerateInputs(2, torch::kBFloat16);
  std::string apprx = "tanh";
  auto expected = torch::gelu_backward(GetCpuInput(1), GetCpuInput(0), apprx);
  auto result = torch::gelu_backward(GetHpuInput(1), GetHpuInput(0), apprx);

  Compare(expected, result, 2e-2, 2e-2);
}
/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_inplace_bfloat16) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::gelu_(GetCpuInput(0));
  auto result = torch::gelu_(GetHpuInput(0));

  Compare(expected, result, 0.01, 0.01);
}

TEST_F(HpuOpTest, gelu_inplace_float) {
  GenerateInputs(1);

  auto expected = torch::gelu_(GetCpuInput(0));
  auto result = torch::gelu_(GetHpuInput(0));

  Compare(expected, result);
}
TEST_F(HpuOpTest, gelu_backwardout_float) {
  GenerateInputs(2, {{4, 5, 6}});
  torch::ScalarType dtype = torch::kFloat32;

  auto expected = torch::empty(0, dtype);

  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::gelu_backward_outf(GetCpuInput(0), GetCpuInput(1), "tanh", expected);
  torch::gelu_backward_outf(GetHpuInput(0), GetHpuInput(1), "tanh", result);

  Compare(expected, result);
}

/*
 * Default tolerance will fail
 * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
 */
TEST_F(HpuOpTest, gelu_backwardout_bfloat) {
  GenerateInputs((2), {{4, 5, 6}}, torch::kBFloat16);
  torch::ScalarType dtype = torch::kBFloat16;

  auto expected = torch::empty(0, dtype);

  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::gelu_backward_outf(GetCpuInput(0), GetCpuInput(1), "none", expected);
  torch::gelu_backward_outf(GetHpuInput(0), GetHpuInput(1), "none", result);

  Compare(expected, result, 1e-2, 1e-2);
}
