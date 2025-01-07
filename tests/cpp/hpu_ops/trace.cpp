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

TEST_F(HpuOpTest, trace_int) {
  GenerateInputs(1, {{4, 4}}, {torch::kInt});
  auto expected = torch::trace(GetCpuInput(0));
  auto result = torch::trace(GetHpuInput(0));

  Compare(expected, result);
}

TEST_F(HpuOpTest, trace_float) {
  GenerateInputs(1, {{4, 4}}, {torch::kFloat});

  auto expected = torch::trace(GetCpuInput(0));
  auto result = torch::trace(GetHpuInput(0));

  Compare(expected, result);
}