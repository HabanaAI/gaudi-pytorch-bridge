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

TEST_F(HpuOpTest, mse_loss_fwd) {
  GenerateInputs(2, {{3, 5}, {3, 5}});
  int reduction = torch::Reduction::Mean;

  auto expected = torch::mse_loss(GetCpuInput(0), GetCpuInput(1), reduction);
  auto result = torch::mse_loss(GetHpuInput(0), GetHpuInput(1), reduction);
  Compare(expected, result);
}

TEST_F(HpuOpTest, mse_loss_bwd) {
  if (isGaudi3()) {
    GTEST_SKIP() << "Test skipped on Gaudi3.";
  }
  GenerateInputs(3, {{16}, {16}, {16}});
  int reduction = torch::Reduction::None;

  auto expected = torch::mse_loss_backward(
      GetCpuInput(2), GetCpuInput(0), GetCpuInput(1), reduction);
  auto result = torch::mse_loss_backward(
      GetCpuInput(2), GetHpuInput(0), GetHpuInput(1), reduction);
  Compare(expected, result);
}

TEST_F(HpuOpTest, mse_loss_fwd_out) {
  GenerateInputs(2, {{3, 2, 8}, {3, 2, 8}});
  int reduction = torch::Reduction::None;

  auto expected = torch::empty(0);
  auto result = expected.to("hpu");

  torch::mse_loss_outf(GetCpuInput(0), GetCpuInput(1), reduction, expected);
  torch::mse_loss_outf(GetHpuInput(0), GetHpuInput(1), reduction, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, mse_loss_bwd_out) {
  GenerateInputs(3, {{10, 5, 4, 4}, {10, 5, 4, 4}, {1}});
  int reduction = torch::Reduction::Sum;

  auto expected = torch::empty(0);
  auto result = expected.to("hpu");

  torch::mse_loss_backward_outf(
      GetCpuInput(2), GetCpuInput(0), GetCpuInput(1), reduction, expected);
  torch::mse_loss_backward_outf(
      GetHpuInput(2), GetHpuInput(0), GetHpuInput(1), reduction, result);
  Compare(expected, result);
}
