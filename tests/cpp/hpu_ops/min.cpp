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
using namespace std;

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, min_keepdim_true) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), 2 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 2 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kBFloat16});
  auto expected = torch::min(GetCpuInput(0), 2 /*dim*/, false /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 2 /*dim*/, false /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_neg_dim_keepdim_true) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), -2 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), -2 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_neg_dim_keepdim_false) {
  GenerateInputs(1, {{2, 3, 4, 5}}, {torch::kBFloat16});
  auto expected = torch::min(GetCpuInput(0), -2 /*dim*/, false /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), -2 /*dim*/, false /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_dim_0_keepdim_true) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), 0 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 0 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}

TEST_F(HpuOpTest, min_dim_0_keepdim_false) {
  GenerateInputs(1);
  auto expected = torch::min(GetCpuInput(0), 0 /*dim*/, true /*keepdim*/);
  auto result = torch::min(GetHpuInput(0), 0 /*dim*/, true /*keepdim*/);

  Compare(get<0>(expected), get<0>(result));
  Compare(get<1>(expected), get<1>(result));
}