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

TEST_F(HpuOpTest, WeightNormTest) {
  GenerateInputs(2, {{16, 32, 64}, {1, 1, 64}}, {at::kFloat, at::kFloat});
  int64_t dim = 2;

  // CPU Run
  at::Tensor expected = at::_weight_norm(GetCpuInput(0), GetCpuInput(1), dim);
  // HPU Run
  at::Tensor result = at::_weight_norm(GetHpuInput(0), GetHpuInput(1), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}

TEST_F(HpuOpTest, WeightNormDim0Test) {
  GenerateInputs(2, {{128, 64}, {128, 1}}, {at::kFloat, at::kFloat});
  int64_t dim = 0;

  // CPU Run
  at::Tensor expected = at::_weight_norm(GetCpuInput(0), GetCpuInput(1), dim);
  // HPU Run
  at::Tensor result = at::_weight_norm(GetHpuInput(0), GetHpuInput(1), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}

TEST_F(HpuOpTest, WeightNormBackwardExecute) {
  GenerateInputs(
      4,
      {{32, 16}, {32, 16}, {32, 1}, {32, 1}},
      {
          at::kFloat,
          at::kFloat,
          at::kFloat,
          at::kFloat,
      });
  int64_t dim = 0;
  // CPU Run
  auto expected = at::_weight_norm_interface_backward(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), GetCpuInput(3), dim);
  // HPU Run
  auto result = at::_weight_norm_interface_backward(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), GetHpuInput(3), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}

TEST_F(HpuOpTest, WeightNormBackwardExecute3D) {
  GenerateInputs(
      4,
      {{32, 4, 5}, {32, 4, 5}, {1, 1, 5}, {1, 1, 5}},
      {
          at::kFloat,
          at::kFloat,
          at::kFloat,
          at::kFloat,
      });
  int64_t dim = 2;
  // CPU Run
  auto expected = at::_weight_norm_interface_backward(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), GetCpuInput(3), dim);
  // HPU Run
  auto result = at::_weight_norm_interface_backward(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), GetHpuInput(3), dim);
  // Compare CPU vs HPU
  Compare(expected, result);
}
