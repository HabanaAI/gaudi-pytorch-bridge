/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
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
