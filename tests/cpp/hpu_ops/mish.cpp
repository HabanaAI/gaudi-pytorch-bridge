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

TEST_F(HpuOpTest, mish) {
  GenerateInputs(1);
  auto expected = torch::mish(GetCpuInput(0));
  auto result = torch::mish(GetHpuInput(0));
  Compare(expected, result);
}

TEST_F(HpuOpTest, mish_) {
  GenerateInputs(1);
  torch::mish_(GetCpuInput(0));
  torch::mish_(GetHpuInput(0));
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, mish_out) {
  GenerateInputs(1);
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::mish_outf(GetCpuInput(0), expected);
  torch::mish_outf(GetHpuInput(0), result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, mish_backward) {
  GenerateInputs(2);
  auto expected = torch::mish_backward(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::mish_backward(GetHpuInput(0), GetHpuInput(1));
  Compare(expected, result);
}
