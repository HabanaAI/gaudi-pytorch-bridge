/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
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

INSTANTIATE_TEST_SUITE_P(
    l1loss,
    L1lossHpuOpTest,
    testing::Values(
        at::Reduction::None,
        at::Reduction::Mean,
        at::Reduction::Sum));
