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

TEST_F(HpuOpTest, trace_int) {
  GenerateInputs(1, {{4, 4}}, {torch::kInt});
  auto expected = torch::trace(GetCpuInput(0));
  auto result = torch::trace(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, trace_float) {
  GenerateInputs(1, {{4, 4}}, {torch::kFloat});

  auto expected = torch::trace(GetCpuInput(0));
  auto result = torch::trace(GetHpuInput(0));

  Compare(expected, result);
}