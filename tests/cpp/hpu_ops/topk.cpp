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

TEST_F(HpuOpTest, topk) {
  GenerateInputs(1, {{10, 3, 2}});
  auto k = 3;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;

  auto expected = torch::topk(GetCpuInput(0), k, dim, largest, sorted);
  auto result = torch::topk(GetHpuInput(0), k, dim, largest, sorted);

  Compare(std::get<0>(expected), std::get<0>(result));
  Compare(std::get<1>(expected), std::get<1>(result));
}

TEST_F(HpuOpTest, topk_out) {
  GenerateInputs(1, {{8, 24, 24, 3}});
  auto k = 5;
  auto dim = 0;
  bool sorted = true;
  bool largest = true;
  auto result = torch::empty(0);
  auto result_h = result.to("hpu");
  auto indices = torch::empty(0).to(torch::kLong);
  auto indices_h = indices.to("hpu");

  torch::topk_outf(GetCpuInput(0), k, dim, largest, sorted, result, indices);
  torch::topk_outf(
      GetHpuInput(0), k, dim, largest, sorted, result_h, indices_h);

  Compare(result, result_h);
  Compare(indices, indices_h);
}