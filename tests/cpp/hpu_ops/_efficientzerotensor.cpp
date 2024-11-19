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

TEST_F(HpuOpTest, EfficientZeroTensor) {
  auto tensor = at::_efficientzerotensor(
      {{3, 2, 3}},
      torch::kFloat32,
      c10::nullopt,
      c10::Device(c10::DeviceType::HPU),
      c10::nullopt);
  auto expected_tensor = at::_efficientzerotensor(
      {{3, 2, 3}}, at::TensorOptions().dtype(torch::kFloat32));

  Compare(expected_tensor, tensor, 0, 0);
}
