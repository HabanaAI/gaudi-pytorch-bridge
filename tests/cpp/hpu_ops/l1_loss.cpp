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

class L1lossHpuOpTest : public HpuOpTestUtil,
                        public testing::WithParamInterface<int64_t> {};
// reduce_mean_fwd doesn't support int
TEST_P(L1lossHpuOpTest, l1_loss) {
  GenerateInputs(2);
  const auto& reduction = GetParam();
  auto expected = torch::l1_loss(GetCpuInput(0), GetCpuInput(1), reduction);
  auto result = torch::l1_loss(GetHpuInput(0), GetHpuInput(1), reduction);
  Compare(expected, result);
}

#if IS_PYTORCH_OLDER_THAN(1, 13)

TEST_P(L1lossHpuOpTest, l1_loss_out) {
  GenerateInputs(3, {torch::kBFloat16});
  torch::ScalarType dtype = torch::kBFloat16;
  const auto& reduction = GetParam();
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::l1_loss_outf(GetCpuInput(0), GetCpuInput(0), reduction, expected);
  torch::l1_loss_outf(GetHpuInput(0), GetHpuInput(0), reduction, result);
  Compare(expected, result);
}

TEST_P(L1lossHpuOpTest, l1_loss_backward) {
  GenerateInputs(3);
  const auto& reduction = GetParam();
  auto expected = torch::l1_loss_backward(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), reduction);
  auto result = torch::l1_loss_backward(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), reduction);
  Compare(expected, result);
}

TEST_P(L1lossHpuOpTest, l1_loss_backward_out) {
  GenerateInputs(3, {torch::kBFloat16});
  const auto& reduction = GetParam();
  torch::ScalarType dtype = torch::kBFloat16;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::l1_loss_backward_outf(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), reduction, expected);
  torch::l1_loss_backward_outf(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), reduction, result);
  Compare(expected, result);
}

#endif

INSTANTIATE_TEST_SUITE_P(
    l1loss,
    L1lossHpuOpTest,
    testing::Values(
        at::Reduction::None,
        at::Reduction::Mean,
        at::Reduction::Sum));
