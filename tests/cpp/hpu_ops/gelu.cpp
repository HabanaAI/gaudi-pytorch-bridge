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

TEST_F(HpuOpTest, gelu_backwardFloat) {
  GenerateInputs(2);

  auto expected = torch::gelu_backward(GetCpuInput(1), GetCpuInput(0));
  auto result = torch::gelu_backward(GetHpuInput(1), GetHpuInput(0));

  Compare(expected, result, 2e-2, 2e-2);
  /*
   * Default tolerance will fail
   * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
   */
}

TEST_F(HpuOpTest, gelu_backwardBFloat16) {
  GenerateInputs(2, torch::kBFloat16);

  auto expected = torch::gelu_backward(GetCpuInput(1), GetCpuInput(0));
  auto result = torch::gelu_backward(GetHpuInput(1), GetHpuInput(0));

  Compare(expected, result, 2e-2, 2e-2);
  /*
   * Default tolerance will fail
   * Issue Raised: https://jira.habana-labs.com/browse/SW-68856
   */
}