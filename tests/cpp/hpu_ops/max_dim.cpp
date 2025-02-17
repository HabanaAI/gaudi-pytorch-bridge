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

TEST_F(HpuOpTest, max_out_keepdim_false) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;
  auto max = torch::empty(0, dtype);
  auto hmax = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto max_values = torch::empty(0, torch::kInt64);
  auto hmax_values = max_values.to("hpu");

  torch::max_outf(GetCpuInput(0), 2, /* keepdim */ false, max, max_values);
  torch::max_outf(GetHpuInput(0), 2, /* keepdim */ false, hmax, hmax_values);
  Compare(max, hmax, 0, 0);
  Compare(max_values, hmax_values, 0, 0);
}

TEST_F(HpuOpTest, max_out_keepdim_true) {
  GenerateInputs(1);

  torch::ScalarType dtype = torch::kFloat;
  auto max = torch::empty(0, dtype);
  auto hmax = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto max_values = torch::empty(0, torch::kInt64);
  auto hmax_values = max_values.to("hpu");

  torch::max_outf(GetCpuInput(0), -2, /* keepdim */ true, max, max_values);
  torch::max_outf(GetHpuInput(0), -2, /* keepdim */ true, hmax, hmax_values);
  Compare(max, hmax, 0, 0);
  Compare(max_values, hmax_values, 0, 0);
}

TEST_F(HpuOpTest, max_out_bf16) {
  GenerateInputs(1, {torch::kBFloat16});

  torch::ScalarType dtype = torch::kBFloat16;
  auto max = torch::empty(0, dtype);
  auto hmax = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto max_values = torch::empty(0, torch::kInt64);
  auto hmax_values = max_values.to("hpu");

  torch::max_outf(GetCpuInput(0), -1, /* keepdim */ true, max, max_values);
  torch::max_outf(GetHpuInput(0), -1, /* keepdim */ true, hmax, hmax_values);
  Compare(max, hmax, 0, 0);
  Compare(max_values, hmax_values, 0, 0);
}

TEST_F(HpuOpTest, max_out_int) {
  GenerateInputs(1, {torch::kInt});

  torch::ScalarType dtype = torch::kInt;
  auto max = torch::empty(0, dtype);
  auto hmax = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto max_values = torch::empty(0, torch::kInt64);
  auto hmax_values = max_values.to("hpu");

  torch::max_outf(GetCpuInput(0), -1, /* keepdim */ false, max, max_values);
  torch::max_outf(GetHpuInput(0), -1, /* keepdim */ false, hmax, hmax_values);
  Compare(max, hmax, 0, 0);
  Compare(max_values, hmax_values, 0, 0);
}

TEST_F(HpuOpTest, max_dim_false) {
  GenerateInputs(1);
  auto exp = torch::max(GetCpuInput(0), 2, /* keepdim */ false);
  auto res = torch::max(GetHpuInput(0), 2, /* keepdim */ false);
  Compare(std::get<0>(exp), std::get<0>(res), 0, 0);
  Compare(std::get<1>(exp), std::get<1>(res), 0, 0);
}

TEST_F(HpuOpTest, max_dim_true) {
  GenerateInputs(1);
  auto exp = torch::max(GetCpuInput(0), 2, /* keepdim */ true);
  auto res = torch::max(GetHpuInput(0), 2, /* keepdim */ true);
  Compare(std::get<0>(exp), std::get<0>(res), 0, 0);
  Compare(std::get<1>(exp), std::get<1>(res), 0, 0);
}

TEST_F(HpuOpTest, max_dim_int) {
  GenerateInputs(1, {torch::kInt});
  auto exp = torch::max(GetCpuInput(0), -1, /* keepdim */ true);
  auto res = torch::max(GetHpuInput(0), -1, /* keepdim */ true);
  Compare(std::get<0>(exp), std::get<0>(res), 0, 0);
  Compare(std::get<1>(exp), std::get<1>(res), 0, 0);
}

TEST_F(HpuOpTest, max_dim_bf16) {
  GenerateInputs(1, {torch::kBFloat16});
  auto exp = torch::max(GetCpuInput(0), 1, /* keepdim */ false);
  auto res = torch::max(GetHpuInput(0), 1, /* keepdim */ false);
  Compare(std::get<0>(exp), std::get<0>(res), 0, 0);
  Compare(std::get<1>(exp), std::get<1>(res), 0, 0);
}