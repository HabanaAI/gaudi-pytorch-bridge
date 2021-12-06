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

TEST_F(HpuOpTest, logaddexp2) {
  GenerateInputs(2);
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::logaddexp2(GetCpuInput(0), GetCpuInput(1));
  auto result = torch::logaddexp2(GetHpuInput(0), GetHpuInput(1));

  Compare(expected, result);
}

TEST_F(HpuOpTest, logaddexp2_out) {
  GenerateInputs(2);
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::logaddexp2_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::logaddexp2_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}
