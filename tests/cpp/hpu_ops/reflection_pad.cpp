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

TEST_F(HpuOpTest, reflection_pad1d) {
  GenerateInputs(1, {{3, 3}});
  // tpc expects the pad array to have the values in the order pad_before
  // for each dim followed by pad_after for each dim
  std::vector<int64_t> pad_size = {{2, 1}};
  auto expected = torch::reflection_pad1d(GetCpuInput(0), pad_size);
  auto result = torch::reflection_pad1d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

// reflection_pad1d cpu not support bf16, int, i16
TEST_F(HpuOpTest, reflection_pad1d_bf16) {
  GenerateInputs(1, {{3, 3}}, {torch::kBFloat16});
  // tpc expects the pad array to have the values in the order pad_before
  // for each dim followed by pad_after for each dim
  std::vector<int64_t> pad_size = {{2, 1}};
  auto expected =
      torch::reflection_pad1d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad1d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kBFloat16), result);
}

TEST_F(HpuOpTest, reflection_pad1d_int) {
  GenerateInputs(1, {{3, 3}}, {torch::kInt});
  // tpc expects the pad array to have the values in the order pad_before
  // for each dim followed by pad_after for each dim
  std::vector<int64_t> pad_size = {{2, 1}};
  auto expected =
      torch::reflection_pad1d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad1d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kInt), result);
}

TEST_F(HpuOpTest, reflection_pad1d_i16) {
  GenerateInputs(1, {{3, 3}}, {torch::kShort});
  // tpc expects the pad array to have the values in the order pad_before
  // for each dim followed by pad_after for each dim
  std::vector<int64_t> pad_size = {{2, 1}};
  auto expected =
      torch::reflection_pad1d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad1d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kShort), result);
}

TEST_F(HpuOpTest, reflection_pad1d_out) {
  GenerateInputs(1, {{3, 4, 5}});
  std::vector<int64_t> pad_size = {{2, 2}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::reflection_pad1d_outf(GetCpuInput(0), pad_size, expected);
  torch::reflection_pad1d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad2d) {
  GenerateInputs(1, {{4, 3, 3, 4}}, {torch::kFloat});
  std::vector<int64_t> pad_size = {{3, 1, 1, 2}};
  auto expected = torch::reflection_pad2d(GetCpuInput(0), pad_size);
  auto result = torch::reflection_pad2d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

// reflection_pad2d cpu not support bf16, int, i16
TEST_F(HpuOpTest, reflection_pad2d_bf16) {
  GenerateInputs(1, {{4, 3, 3, 4}}, {torch::kBFloat16});
  std::vector<int64_t> pad_size = {{3, 1, 1, 2}};
  auto expected =
      torch::reflection_pad2d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad2d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kBFloat16), result);
}

TEST_F(HpuOpTest, reflection_pad2d_int) {
  GenerateInputs(1, {{4, 3, 3, 4}}, {torch::kInt});
  std::vector<int64_t> pad_size = {{3, 1, 1, 2}};
  auto expected =
      torch::reflection_pad2d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad2d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kInt), result);
}

TEST_F(HpuOpTest, reflection_pad2d_i16) {
  GenerateInputs(1, {{4, 3, 3, 4}}, {torch::kShort});
  std::vector<int64_t> pad_size = {{3, 1, 1, 2}};
  auto expected =
      torch::reflection_pad2d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad2d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kShort), result);
}

TEST_F(HpuOpTest, reflection_pad2d_out) {
  GenerateInputs(1, {{5, 4, 4, 5}}, {torch::kFloat});
  std::vector<int64_t> pad_size = {{3, 2, 2, 3}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::reflection_pad2d_outf(GetCpuInput(0), pad_size, expected);
  torch::reflection_pad2d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad1d_backward) {
  GenerateInputs(2, {{2, 2}, {2, 4}});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{1, 1}};
  auto expected = torch::reflection_pad1d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result =
      torch::reflection_pad1d_backward(hgrad_out, GetHpuInput(0), pad_size);
  Compare(expected, result);
}

// reflection_pad1d cpu not support int
TEST_F(HpuOpTest, reflection_pad1d_backward_int) {
  GenerateInputs(2, {{2, 2}, {2, 4}}, {torch::kInt, torch::kInt});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{1, 1}};
  auto expected = torch::reflection_pad1d_backward(
      GetCpuInput(1).to(torch::kFloat),
      GetCpuInput(0).to(torch::kFloat),
      pad_size);
  auto result =
      torch::reflection_pad1d_backward(hgrad_out, GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kInt), result);
}

TEST_F(HpuOpTest, reflection_pad1d_backward_out) {
  GenerateInputs(2, {{2, 3, 2}, {2, 3, 4}});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{1, 1}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::reflection_pad1d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::reflection_pad1d_backward_outf(
      hgrad_out, GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad2d_backward) {
  GenerateInputs(2, {{2, 3, 2, 4}, {2, 3, 3, 7}});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{1, 2, 1, 0}};
  auto expected = torch::reflection_pad2d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result =
      torch::reflection_pad2d_backward(hgrad_out, GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad2d_backward_out) {
  GenerateInputs(2, {{2, 3, 2}, {2, 3, 4}});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{1, 1, 0, 0}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::reflection_pad2d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::reflection_pad2d_backward_outf(
      hgrad_out, GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad3d) {
  GenerateInputs(1, {{1, 2, 3, 2, 4}});
  std::vector<int64_t> pad_size = {{3, 3, 1, 1, 2, 2}};
  auto expected = torch::reflection_pad3d(GetCpuInput(0), pad_size);
  auto result = torch::reflection_pad3d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

// reflection_pad3d cpu not support bf16, int, i16
TEST_F(HpuOpTest, reflection_pad3d_int) {
  GenerateInputs(1, {{1, 2, 3, 2, 4}}, {torch::kInt});
  std::vector<int64_t> pad_size = {{3, 3, 1, 1, 2, 2}};
  auto expected =
      torch::reflection_pad3d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad3d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kInt), result);
}

TEST_F(HpuOpTest, reflection_pad3d_bf16) {
  GenerateInputs(1, {{1, 2, 3, 2, 4}}, {torch::kBFloat16});
  std::vector<int64_t> pad_size = {{3, 3, 1, 1, 2, 2}};
  auto expected =
      torch::reflection_pad3d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad3d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kBFloat16), result);
}

TEST_F(HpuOpTest, reflection_pad3d_i16) {
  GenerateInputs(1, {{1, 2, 3, 2, 4}}, {torch::kShort});
  std::vector<int64_t> pad_size = {{3, 3, 1, 1, 2, 2}};
  auto expected =
      torch::reflection_pad3d(GetCpuInput(0).to(torch::kFloat), pad_size);
  auto result = torch::reflection_pad3d(GetHpuInput(0), pad_size);
  Compare(expected.to(torch::kShort), result);
}

TEST_F(HpuOpTest, reflection_pad3d_out) {
  GenerateInputs(1, {{3, 3, 3, 4}});
  std::vector<int64_t> pad_size = {{1, 1, 2, 2, 1, 1}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::reflection_pad3d_outf(GetCpuInput(0), pad_size, expected);
  torch::reflection_pad3d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad3d_backward) {
  GenerateInputs(2, {{1, 2, 3, 2, 4}, {1, 2, 7, 4, 10}});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{3, 3, 1, 1, 2, 2}};
  auto expected = torch::reflection_pad3d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result =
      torch::reflection_pad3d_backward(hgrad_out, GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, reflection_pad3d_backward_out) {
  GenerateInputs(2, {{2, 3, 4, 5}, {2, 5, 4, 7}});
  auto hgrad_out = GetCpuInput(1).to(torch::kHPU);
  std::vector<int64_t> pad_size = {{1, 1, 0, 0, 1, 1}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::reflection_pad3d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::reflection_pad3d_backward_outf(
      hgrad_out, GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}