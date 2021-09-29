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

TEST_F(HpuOpTest, logcumsumexp_out) {
  GenerateInputs(1, torch::kFloat);
  int64_t dim = 1;
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::logcumsumexp_outf(GetCpuInput(0), dim, expected);
  torch::logcumsumexp_outf(GetHpuInput(0), dim, result);

  Compare(expected, result);
}
