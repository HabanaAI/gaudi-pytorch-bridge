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

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, channel_shuffle) {
  GenerateInputs(1, {{1, 8, 4, 4}}, {torch::kInt});

  auto expected = torch::channel_shuffle(GetCpuInput(0), 2 /*groups*/);
  auto result = torch::channel_shuffle(GetHpuInput(0), 2 /*groups*/);
  // tol set to 0 since there is no computation.
  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, channel_shuffle_group1) {
  GenerateInputs(1, {{64, 64, 128}}, {torch::kBFloat16});

  auto expected = torch::channel_shuffle(GetCpuInput(0), 1 /*groups*/);
  auto result = torch::channel_shuffle(GetHpuInput(0), 1 /*groups*/);

  Compare(expected, result, 0, 0);
}

TEST_F(HpuOpTest, channel_shuffle_group3) {
  GenerateInputs(1, {{1, 3, 4, 5, 6}}, {torch::kFloat});

  auto expected = torch::channel_shuffle(GetCpuInput(0), 1 /*groups*/);
  auto result = torch::channel_shuffle(GetHpuInput(0), 1 /*groups*/);

  Compare(expected, result, 0, 0);
}