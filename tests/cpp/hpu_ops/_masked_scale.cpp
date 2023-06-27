/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
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

TEST_F(HpuOpTest, _masked_scale1) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(2, {{28}, {28}});
  float scale = 5.6;

  auto expected = at::_masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  auto result = _masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  Compare(expected, result);
}

TEST_F(HpuOpTest, _masked_scale2) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(2, {{1, 1, 8}, {2, 2, 8}}, {torch::kBFloat16});
  float scale = 0.6;

  auto expected = at::_masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  auto result = _masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  Compare(expected, result);
}