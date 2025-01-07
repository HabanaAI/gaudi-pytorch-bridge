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

TEST_F(HpuOpTest, hardswish) {
  GenerateInputs(1, {{2, 2}}, {torch::kFloat});

  auto hresult = torch::hardswish(GetHpuInput(0));
  auto cout = torch::hardswish(GetCpuInput(0));

  Compare(cout, hresult);
}

TEST_F(HpuOpTest, hardswish_) {
  GenerateInputs(1, {{2, 2}}, {torch::kFloat});

  auto hresult = torch::hardswish_(GetHpuInput(0));
  auto cout = torch::hardswish_(GetCpuInput(0));

  Compare(cout, hresult);
}

TEST_F(HpuOpTest, hardswish_out) {
  GenerateInputs(1, {{2, 2}}, {torch::kFloat});

  auto hresult =
      torch::empty(0, torch::TensorOptions(torch::kFloat).device("hpu"));
  auto cout = torch::empty(0, torch::kFloat);

  hresult = torch::hardswish_outf(GetHpuInput(0), hresult);
  cout = torch::hardswish_outf(GetCpuInput(0), cout);

  Compare(cout, hresult);
}

TEST_F(HpuOpTest, hardswish_backward) {
  GenerateInputs(2, {{2, 2}}, {torch::kFloat});

  auto hresult = torch::hardswish_backward(GetHpuInput(0), GetHpuInput(1));
  auto cout = torch::hardswish_backward(GetCpuInput(0), GetCpuInput(1));

  Compare(cout, hresult);
}
