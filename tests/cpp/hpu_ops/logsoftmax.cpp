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

TEST_F(HpuOpTest, log_softmax_int) {
  GenerateInputs(1);
  int64_t dim = -1;
  auto exp = torch::log_softmax(GetCpuInput(0), dim);
  auto res = torch::log_softmax(GetHpuInput(0), dim);
  Compare(exp, res);
}

TEST_F(HpuOpTest, log_softmax_int_bf16) {
  GenerateInputs(1, {torch::kBFloat16});
  int64_t dim = 1;
  auto exp = torch::log_softmax(GetCpuInput(0), dim);
  auto res = torch::log_softmax(GetHpuInput(0), dim);
  Compare(exp, res, 0.01, 0.01);
}