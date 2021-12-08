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

// sinc usual variant with 0 included in the test case
TEST_F(HpuOpTest, sinc) {
  GenerateInputs(1, {{12, 16, 24, 0}});

  auto expected = torch::sinc(GetCpuInput(0));
  auto result = torch::sinc(GetHpuInput(0));

  Compare(expected, result);
}

// sinc out variant for float datatype
TEST_F(HpuOpTest, sinc_out) {
  GenerateInputs(1, {{2, 3}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::sinc_outf(GetCpuInput(0), expected);
  torch::sinc_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

// sinc inplace variant
TEST_F(HpuOpTest, sinc_) {
  GenerateInputs(1);

  torch::sinc_(GetCpuInput(0));
  torch::sinc_(GetHpuInput(0));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

// sinc inplace variant for bf16 datatype
TEST_F(HpuOpTest, sinc_bf16) {
  GenerateInputs(1, torch::kBFloat16);

  torch::sinc_(GetCpuInput(0));
  torch::sinc_(GetHpuInput(0));

  Compare(GetCpuInput(0), GetHpuInput(0), 0.32, 1e-3);
}