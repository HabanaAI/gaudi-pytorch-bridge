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

TEST_F(HpuOpTest, new_zeros) {
  GenerateIntInputs(1, {{10, 3, 2}}, 0, 100);
  torch::IntArrayRef targetShape = {2, 4, 5};
  auto cpuTensor = GetCpuInput(0).to(torch::kLong);
  auto hpuTensor = GetHpuInput(0).to(torch::kLong);
  auto expected = at::native::new_zeros(cpuTensor, targetShape);
  auto result = at::native::new_zeros(hpuTensor, targetShape);
  Compare(expected, result);
}

TEST_F(HpuOpTest, new_zeros_with_dtype) {
  GenerateInputs(1, {{10, 3, 2}}, {torch::kFloat32});
  torch::IntArrayRef targetShape = {2, 4, 5};
  auto cpuTensor = GetCpuInput(0);
  auto hpuTensor = GetHpuInput(0);
  auto targetType = torch::kBFloat16;
  auto expected = at::native::new_zeros(cpuTensor, targetShape, targetType);
  auto result = at::native::new_zeros(hpuTensor, targetShape, targetType);
  Compare(expected, result);
}