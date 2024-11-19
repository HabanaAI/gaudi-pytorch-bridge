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

TEST_F(HpuOpTest, log_softmax_int) {
  GenerateInputs(1);
  int64_t dim = -1;
  auto exp = torch::log_softmax(GetCpuInput(0), dim);
  auto res = torch::log_softmax(GetHpuInput(0), dim);
  Compare(exp, res);
}

TEST_F(HpuOpTest, log_softmax_int_bf16) {
  GenerateInputs(1, {torch::kBFloat16});
  int64_t dim = 1;
  auto exp = torch::log_softmax(GetCpuInput(0), dim);
  auto res = torch::log_softmax(GetHpuInput(0), dim);
  Compare(exp, res, 0.01, 0.01);
}