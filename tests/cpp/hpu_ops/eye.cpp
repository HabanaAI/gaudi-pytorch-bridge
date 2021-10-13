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

TEST_F(HpuOpTest, eye_out) {
  int64_t n = 3;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({n, n}, dtype);
  auto result = torch::empty({n, n}, torch::TensorOptions(dtype).device("hpu"));

  torch::eye_outf(n, expected);
  torch::eye_outf(n, result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, eye_m_out) {
  int64_t n = 3;
  int64_t m = 2;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty({n, m}, dtype);
  auto result = torch::empty({n, m}, torch::TensorOptions(dtype).device("hpu"));

  torch::eye_outf(n, m, expected);
  torch::eye_outf(n, m, result);

  Compare(expected, result);
}
