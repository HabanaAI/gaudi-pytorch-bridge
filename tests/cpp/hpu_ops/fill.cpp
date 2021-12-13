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

TEST_F(HpuOpTest, fill_tensor_inplace_float) {
  GenerateInputs(2, {{3, 2, 3}, {}});

  GetCpuInput(0).fill_(GetCpuInput(1));
  GetHpuInput(0).fill_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, fill_tensor_inplace_bfloat) {
  GenerateInputs(2, {{4, 5, 1, 2}, {}}, torch::kBFloat16);

  GetCpuInput(0).fill_(GetCpuInput(1));
  GetHpuInput(0).fill_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, fill_tensor_inplace_int) {
  GenerateInputs(2, {{6, 8}, {}}, torch::kInt32);

  GetCpuInput(0).fill_(GetCpuInput(1));
  GetHpuInput(0).fill_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}
