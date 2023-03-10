/******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "util.h"
class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, sort) {
  GenerateInputs(1, {{10, 3, 2}});
  auto k = 3;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;

  auto expected = torch::sort(GetCpuInput(0), dim, largest);
  auto result = torch::sort(GetHpuInput(0), dim, largest);

  Compare(std::get<0>(expected), std::get<0>(result));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, sort_out) {
  GenerateInputs(1, {{8, 24, 24, 3}});
  int dim = 2;
  c10::optional<bool> stable(false);
  bool descending = false;
  auto result = torch::empty(0);
  auto result_h = result.to("hpu");
  auto indices = torch::empty(0).to(torch::kLong);
  auto indices_h = indices.to("hpu");

  torch::sort_outf(GetCpuInput(0), stable, dim, descending, result, indices);
  torch::sort_outf(
      GetHpuInput(0), stable, dim, descending, result_h, indices_h);
  Compare(result, result_h);
  Compare(indices, indices_h);
}
