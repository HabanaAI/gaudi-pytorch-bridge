/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "../utils/device_type_util.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, linspace) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  // Not supporting for the values beyond 40
  float start = GenerateScalar<float>(-1, 40);
  // Not supporting for the values use beyond 40
  float end = GenerateScalar<float>(1, 40);
  int steps = GenerateScalar<int>();
  auto expected = torch::linspace(start, end, steps);
  auto result = torch::linspace(start, end, steps, "hpu");
  Compare(expected, result);
}
