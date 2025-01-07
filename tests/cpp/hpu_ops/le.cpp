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

TEST_F(HpuOpTest, le_scalar) {
  GenerateInputs(1, {{2, 3, 4}});
  float other = 1.1;

  GetCpuInput(0).le_(other);
  GetHpuInput(0).le_(other);

  Compare(GetCpuInput(0), GetHpuInput(0));
}

TEST_F(HpuOpTest, le_tensor) {
  GenerateInputs(2, {{2, 3, 4}, {2, 3, 4}});

  GetCpuInput(0).le_(GetCpuInput(1));
  GetHpuInput(0).le_(GetHpuInput(1));

  Compare(GetCpuInput(0), GetHpuInput(0));
}
