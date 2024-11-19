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

TEST_F(HpuOpTest, linspace) {
  // Not supporting for the values beyond 40
  float start = GenerateScalar<float>(-1, 40);
  // Not supporting for the values use beyond 40
  float end = GenerateScalar<float>(1, 40);
  int steps = GenerateScalar<int>();
  auto expected = torch::linspace(start, end, steps);
  auto result = torch::linspace(start, end, steps, "hpu");
  Compare(expected, result);
}
