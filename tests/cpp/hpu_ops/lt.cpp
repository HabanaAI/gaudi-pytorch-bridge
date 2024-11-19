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

#include <stdexcept>
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, lt_scalar_) {
  GenerateInputs(1, torch::kInt);
  float compVal = -0.1f;

  GetCpuInput(0).lt_(compVal);
  GetHpuInput(0).lt_(compVal);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, lt_tensor_) {
  GenerateInputs(2, torch::kInt32);

  GetCpuInput(0).lt_(GetCpuInput(0.));
  GetHpuInput(0).lt_(GetHpuInput(0.));

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, lt_scalar) {
  GenerateInputs(1, torch::kInt);
  float compVal = 0;

  auto exp = GetCpuInput(0).lt(compVal);
  auto res = GetHpuInput(0).lt(compVal);

  Compare(exp, res);
}
