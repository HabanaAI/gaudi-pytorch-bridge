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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, cat) {
  GenerateInputs(3, {at::kFloat, at::kInt, at::kBFloat16});
  auto expected = torch::cat({GetCpuInput(0), GetCpuInput(1), GetCpuInput(2)});
  auto result = torch::cat({GetHpuInput(0), GetHpuInput(1), GetHpuInput(2)});
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, cat_empty) {
  GenerateInputs(2, {{0}, {8, 3, 24, 24}});
  auto expected = torch::cat({GetCpuInput(0), GetCpuInput(1)}, 1);
  auto result = torch::cat({GetHpuInput(0), GetHpuInput(1)}, 1);
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, DISABLED_cat_out) {
  // GenerateInputs(3, {at::kFloat, at::kInt, at::kBFloat16});
  GenerateInputs(3, {at::kInt, at::kBFloat16, at::kBFloat16});
  auto expected = torch::empty(0, at::kBFloat16);
  auto result = expected.to(at::kHPU);
  torch::cat_outf(
      {GetCpuInput(0), GetCpuInput(1), GetCpuInput(2)}, 0, expected);
  torch::cat_outf({GetHpuInput(0), GetHpuInput(1), GetHpuInput(2)}, 0, result);

  Compare(expected, result, 0, 0);
}
