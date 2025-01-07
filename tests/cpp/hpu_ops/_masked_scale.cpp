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

TEST_F(HpuOpTest, _masked_scale1) {
  GenerateInputs(2, {{28}, {28}});
  float scale = 5.6;

  auto expected = at::_masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  auto result = _masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  Compare(expected, result);
}

TEST_F(HpuOpTest, _masked_scale2) {
  GenerateInputs(2, {{1, 1, 8}, {1, 1, 8}}, {torch::kBFloat16});
  float scale = 0.6;

  auto expected = at::_masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  auto result = _masked_scale(GetHpuInput(0), GetHpuInput(1), scale);
  Compare(expected, result);
}