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

TEST_F(HpuOpTest, Relu) {
  GenerateInputs(1);

  auto expected = torch::relu(GetCpuInput(0));
  auto result = torch::relu(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, Relu_) {
  GenerateInputs(1);

  auto expected = GetCpuInput(0).relu();
  auto result = GetHpuInput(0).relu();

  Compare(expected, result);
}

TEST_F(HpuOpTest, Relu_out) {
  GenerateInputs(1, torch::kBFloat16);

  auto expected = torch::relu(GetCpuInput(0));
  auto result = torch::empty(0).to(torch::kBFloat16).to("hpu");
  torch::relu_outf(GetHpuInput(0), result);

  Compare(expected, result);
}
