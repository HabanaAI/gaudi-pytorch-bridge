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
using namespace std;

class HpuOpTest : public HpuOpTestUtil {};

TEST_F(HpuOpTest, replication_pad1d_input2d) {
  GenerateInputs(1, {{2, 3}});
  std::vector<int64_t> pad_size = {{3, 1}};
  auto expected = torch::replication_pad1d(GetCpuInput(0), pad_size);
  auto result = torch::replication_pad1d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad1d_input3d) {
  GenerateInputs(1, {{2, 3, 4}});
  std::vector<int64_t> pad_size = {{1, 2}};
  auto expected = torch::replication_pad1d(GetCpuInput(0), pad_size);
  auto result = torch::replication_pad1d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_input3d) {
  GenerateInputs(1, {{2, 3, 4}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5}};
  auto expected = torch::replication_pad2d(GetCpuInput(0), pad_size);
  auto result = torch::replication_pad2d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_input4d) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5}};
  auto expected = torch::replication_pad2d(GetCpuInput(0), pad_size);
  auto result = torch::replication_pad2d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_input4d) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5, 6, 7}};
  auto expected = torch::replication_pad3d(GetCpuInput(0), pad_size);
  auto result = torch::replication_pad3d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_input5d) {
  GenerateInputs(1, {{2, 3, 4, 5, 6}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5, 6, 7}};
  auto expected = torch::replication_pad3d(GetCpuInput(0), pad_size);
  auto result = torch::replication_pad3d(GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad1d_input3d_out) {
  GenerateInputs(1, {{2, 3, 4}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> pad_size = {{1, 2}};
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad1d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad1d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_input4d_out) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad2d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad2d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_input5d_out) {
  GenerateInputs(1, {{2, 3, 4, 5, 6}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5, 6, 7}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad3d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad3d_outf(GetHpuInput(0), pad_size, result);
}

TEST_F(HpuOpTest, replication_pad2d_input4d_out_zero_pad) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  std::vector<int64_t> pad_size = {{0, 0, 0, 0}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad2d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad2d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad1d_backward_2dinput) {
  GenerateInputs(2, {{14, 14}, {14, 31}});
  std::vector<int64_t> pad_size = {{15, 2}};
  auto expected = torch::replication_pad1d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result = torch::replication_pad1d_backward(
      GetHpuInput(1), GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad1d_backward_3dinput) {
  GenerateInputs(2, {{8, 4, 4}, {8, 4, 9}});
  std::vector<int64_t> pad_size = {{4, 1}};
  auto expected = torch::replication_pad1d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result = torch::replication_pad1d_backward(
      GetHpuInput(1), GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_backward_3dinput) {
  GenerateInputs(2, {{1, 8, 8}, {1, 24, 25}});
  std::vector<int64_t> pad_size = {{2, 15, 14, 2}};
  auto expected = torch::replication_pad2d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result = torch::replication_pad2d_backward(
      GetHpuInput(1), GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_backward_4dinput) {
  GenerateInputs(2, {{1, 2, 14, 14}, {1, 2, 23, 19}});
  std::vector<int64_t> pad_size = {{1, 4, 4, 5}};
  auto expected = torch::replication_pad2d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result = torch::replication_pad2d_backward(
      GetHpuInput(1), GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_backward_4dinput) {
  GenerateInputs(2, {{1, 3, 2, 4}, {1, 7, 10, 11}});
  std::vector<int64_t> pad_size = {{2, 5, 6, 2, 3, 1}};
  auto expected = torch::replication_pad3d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result = torch::replication_pad3d_backward(
      GetHpuInput(1), GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_backward_5dinput) {
  GenerateInputs(2, {{1, 1, 2, 2, 4}, {1, 1, 6, 10, 11}});
  std::vector<int64_t> pad_size = {{3, 4, 5, 3, 1, 3}};
  auto expected = torch::replication_pad3d_backward(
      GetCpuInput(1), GetCpuInput(0), pad_size);
  auto result = torch::replication_pad3d_backward(
      GetHpuInput(1), GetHpuInput(0), pad_size);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad1d_backward_2dinput_out) {
  GenerateInputs(2, {{14, 14}, {14, 31}});
  std::vector<int64_t> pad_size = {{15, 2}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad1d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::replication_pad1d_backward_outf(
      GetHpuInput(1), GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad1d_backward_3dinput_out) {
  GenerateInputs(2, {{2, 4, 24}, {2, 4, 29}});
  std::vector<int64_t> pad_size = {{4, 1}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad1d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::replication_pad1d_backward_outf(
      GetHpuInput(1), GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_backward_3dinput_out) {
  GenerateInputs(2, {{1, 6, 6}, {1, 22, 23}});
  std::vector<int64_t> pad_size = {{2, 15, 14, 2}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad2d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::replication_pad2d_backward_outf(
      GetHpuInput(1), GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_backward_4dinput_out) {
  GenerateInputs(2, {{1, 4, 4, 4}, {1, 4, 13, 9}});
  std::vector<int64_t> pad_size = {{1, 4, 4, 5}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad2d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::replication_pad2d_backward_outf(
      GetHpuInput(1), GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_backward_4dinput_out) {
  GenerateInputs(2, {{1, 2, 2, 8}, {1, 6, 10, 15}});
  std::vector<int64_t> pad_size = {{2, 5, 6, 2, 3, 1}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad3d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::replication_pad3d_backward_outf(
      GetHpuInput(1), GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_backward_5dinput_out) {
  GenerateInputs(2, {{1, 1, 2, 2, 6}, {1, 1, 6, 10, 13}});
  std::vector<int64_t> pad_size = {{3, 4, 5, 3, 1, 3}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty(0, dtype);
  auto result = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad3d_backward_outf(
      GetCpuInput(1), GetCpuInput(0), pad_size, expected);
  torch::replication_pad3d_backward_outf(
      GetHpuInput(1), GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}
