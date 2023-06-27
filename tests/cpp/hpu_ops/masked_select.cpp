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

TEST_F(HpuOpTest, masked_select_usual_mix) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(1, {{2, 3, 28, 28}});
  auto mask = torch::ge(GetCpuInput(0), -10); // mask - mix of True and False
  auto maskh = mask.to("hpu");

  auto expected = torch::masked_select(GetCpuInput(0), mask);
  auto result = torch::masked_select(GetHpuInput(0), maskh);
  Compare(expected, result);
}

TEST_F(HpuOpTest, masked_select_usual_all_false) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(1, {{128, 128}}, {torch::kBFloat16});
  auto mask = torch::zeros({128, 128}).to(torch::kBool); // mask - all false
  auto maskh = mask.to("hpu");

  auto expected = torch::masked_select(GetCpuInput(0), mask);
  auto result = torch::masked_select(GetHpuInput(0), maskh);
  Compare(expected, result);
}

TEST_F(HpuOpTest, masked_select_usual_all_true) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(1, {{3, 64, 64}}, {torch::kInt32});
  auto mask = torch::ones({3, 64, 64}).to(torch::kBool); // mask - all true
  auto maskh = mask.to("hpu");

  auto expected = torch::masked_select(GetCpuInput(0), mask);
  auto result = torch::masked_select(GetHpuInput(0), maskh);
  Compare(expected, result);
}

TEST_F(HpuOpTest, masked_select_out_mix) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(1, {{3, 64, 64}});
  auto mask = torch::ge(GetCpuInput(0), 1.56); // mask - mix of True and False
  auto maskh = mask.to("hpu");

  torch::ScalarType dtype = torch::kFloat32;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  torch::masked_select_outf(GetCpuInput(0), mask, expected);
  torch::masked_select_outf(GetHpuInput(0), maskh, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, masked_select_out_all_true) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(1, {{1024}}, {torch::kBFloat16});
  auto mask = torch::ones({1024}).to(torch::kBool); // mask - all true
  auto maskh = mask.to("hpu");

  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty({1, 1}, dtype); // out tensor with non-empty size
  auto result = torch::empty({1, 1}, torch::TensorOptions(dtype).device("hpu"));

  torch::masked_select_outf(GetCpuInput(0), mask, expected);
  torch::masked_select_outf(GetHpuInput(0), maskh, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, masked_select_out_all_false) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(1, {{2, 3, 4, 5, 6}}, {torch::kInt32});
  auto mask =
      torch::zeros({2, 3, 4, 5, 6}).to(torch::kBool); // mask - all false
  auto maskh = mask.to("hpu");

  torch::ScalarType dtype = torch::kInt32;
  auto expected =
      torch::empty({2, 3, 4, 5, 6}, dtype); // out tensor with non-empty size
  auto result =
      torch::empty({2, 3, 4, 5, 6}, torch::TensorOptions(dtype).device("hpu"));

  torch::masked_select_outf(GetCpuInput(0), mask, expected);
  torch::masked_select_outf(GetHpuInput(0), maskh, result);
  Compare(expected, result);
}
