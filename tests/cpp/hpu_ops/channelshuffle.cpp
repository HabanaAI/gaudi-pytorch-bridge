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

TEST_F(HpuOpTest, channel_shuffle) {
  GenerateInputs(1, {{1, 8, 4, 4}}, {torch::kInt});

  auto expected = torch::channel_shuffle(GetCpuInput(0), 2 /*groups*/);
  auto result = torch::channel_shuffle(GetHpuInput(0), 2 /*groups*/);
  // tol set to 0 since there is no computation.
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, channel_shuffle_group1) {
  GenerateInputs(1, {{64, 64, 128}}, {torch::kBFloat16});

  auto expected = torch::channel_shuffle(GetCpuInput(0), 1 /*groups*/);
  auto result = torch::channel_shuffle(GetHpuInput(0), 1 /*groups*/);

  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, channel_shuffle_group3) {
  GenerateInputs(1, {{1, 3, 4, 5, 6}}, {torch::kFloat});

  auto expected = torch::channel_shuffle(GetCpuInput(0), 1 /*groups*/);
  auto result = torch::channel_shuffle(GetHpuInput(0), 1 /*groups*/);

  Compare(expected, result, 0, 0);
}