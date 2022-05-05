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

TEST_F(HpuOpTest, erfinv) {
  GenerateInputs(1);

  auto expected = torch::erfinv(GetCpuInput(0));
  auto result = torch::erfinv(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, erfinv_out) {
  GenerateInputs(1);
  torch::ScalarType dtype = torch::kFloat32;

  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::erfinv_outf(GetCpuInput(0), expected);
  torch::erfinv_outf(GetHpuInput(0), result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, erfinv_) {
  GenerateInputs(1);

  GetCpuInput(0).erfinv_();
  GetHpuInput(0).erfinv_();

  Compare(GetCpuInput(0), GetHpuInput(0));
}