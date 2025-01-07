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

TEST_F(HpuOpTest, cosine_embedding_loss) {
  GenerateInputs(3, {{5, 6}, {5, 6}, {5}});
  float margin = GenerateScalar<float>(0, 0.5);
  int reduction = GenerateScalar<int>(0, 2);
  auto expected = torch::cosine_embedding_loss(
      GetCpuInput(0), GetCpuInput(1), GetCpuInput(2), margin, reduction);
  auto result = torch::cosine_embedding_loss(
      GetHpuInput(0), GetHpuInput(1), GetHpuInput(2), margin, reduction);
  Compare(expected, result);
}

TEST_F(HpuOpTest, cosine_similarity) {
  GenerateInputs(2);
  float eps = GenerateScalar<float>(1e-10, 1e-7);
  int dim = -1;
  auto expected =
      torch::cosine_similarity(GetCpuInput(0), GetCpuInput(1), dim, eps);
  auto result =
      torch::cosine_similarity(GetHpuInput(0), GetHpuInput(1), dim, eps);
  Compare(expected, result);
}