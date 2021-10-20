/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include <gtest/gtest.h>
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, pow_tensor_tensor_out) {
  GenerateInputs(2, {{2, 4}, {2, 4}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({2, 4}, dtype);
  auto result = torch::empty({2, 4}, torch::TensorOptions(dtype).device("hpu"));

  torch::pow_outf(GetCpuInput(0), GetCpuInput(1), expected);
  torch::pow_outf(GetHpuInput(0), GetHpuInput(1), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, pow_out_non_hpu_out_tensor) {
  GenerateInputs(2);
  auto out = torch::empty(0);
  EXPECT_THROW(
      torch::pow_outf(GetHpuInput(0), GetHpuInput(1), out), c10::Error);
}
