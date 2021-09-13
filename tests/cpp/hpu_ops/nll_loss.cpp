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

TEST_F(HpuOpTest, nll_loss_out) {
  GenerateInputs(1, {{4, 5}});
  auto target = torch::tensor({1, 0, 4, 3});
  auto weight = c10::nullopt;
  int reduction = torch::Reduction::None;
  int ignore_index = -100;
  auto out = torch::empty(0);
  auto total_weight = torch::empty(0);
  auto hout = torch::empty(0, c10::kHPU);
  auto htotal_weight = torch::empty(0, c10::kHPU);

  auto exp = torch::nll_loss_forward_outf(
      GetCpuInput(0),
      target,
      weight,
      reduction,
      ignore_index,
      out,
      total_weight);
  auto res = torch::nll_loss_forward_outf(
      GetHpuInput(0),
      target,
      weight,
      reduction,
      ignore_index,
      hout,
      htotal_weight);
  Compare(std::get<0>(exp), std::get<0>(res));
}
