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

TEST_F(HpuOpTest, arange_start_out) {
  constexpr float start = 1.2;
  constexpr float end = 6.3;
  constexpr float step = 0.8;
  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::arange_outf(start, end, step, expected);
  torch::arange_outf(start, end, step, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, arange_out) {
  constexpr float end = 6.0;
  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::arange_outf(end, expected);
  torch::arange_outf(end, result);
  Compare(expected, result);
}