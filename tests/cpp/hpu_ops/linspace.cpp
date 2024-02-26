/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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

TEST_F(HpuOpTest, linspace) {
  // Not supporting for the values beyond 40
  float start = GenerateScalar<float>(-1, 40);
  // Not supporting for the values use beyond 40
  float end = GenerateScalar<float>(1, 40);
  int steps = GenerateScalar<int>();
  auto expected = torch::linspace(start, end, steps);
  auto result = torch::linspace(start, end, steps, "hpu");
  Compare(expected, result);
}
