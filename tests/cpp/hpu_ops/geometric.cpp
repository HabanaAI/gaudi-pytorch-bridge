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

TEST_F(HpuOpTest, geometric_f32) {
  GenerateInputs(1, {{1, 2}}, {torch::kFloat32});
  double p = 0.8;

  GetCpuInput(0).geometric_(p, at::detail::getDefaultCPUGenerator());
  GetHpuInput(0).geometric_(p, at::detail::getDefaultCPUGenerator());
  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, geometric_bf16) {
  GenerateInputs(1, {{1, 2, 4}}, {torch::kBFloat16});
  double p = 0.9;

  GetCpuInput(0).geometric_(p);
  GetHpuInput(0).geometric_(p);
  Compare(GetCpuInput(0), GetHpuInput(0));
}