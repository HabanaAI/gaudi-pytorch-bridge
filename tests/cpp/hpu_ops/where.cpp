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

TEST_F(HpuOpTest, where) {
  GenerateInputs(2, {torch::kInt8, torch::kFloat}, {{7}, {5, 6, 7}});
  auto condition = torch::randn({7}) > 0;

  auto exp = torch::where(condition, GetCpuInput(0), GetCpuInput(1));
  auto res = torch::where(condition.to("hpu"), GetHpuInput(0), GetHpuInput(1));

  Compare(exp, res, 0, 0);
}

TEST_F(HpuOpTest, where_out) {
  GenerateInputs(2, {torch::kInt, torch::kBFloat16}, {{3}, {4, 2, 3}});
  auto condition = torch::randn({2, 3}) > 0;

  auto dtype = torch::kBFloat16;
  auto exp = torch::empty(0, dtype);
  auto res = torch::empty(0, torch::TensorOptions(dtype).device(at::kHPU));

  torch::where_outf(condition, GetCpuInput(0), GetCpuInput(1), exp);
  torch::where_outf(condition.to("hpu"), GetHpuInput(0), GetHpuInput(1), res);

  Compare(exp, res, 0, 0);
}
