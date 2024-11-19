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

TEST_F(HpuOpTest, avg_pool2d_3d_f32) {
  GenerateInputs(1, {{4, 7, 5}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::avg_pool2d(
      GetCpuInput(0), kernel_size, stride, pad, ceil, count_include_pad, {});
  auto result = torch::avg_pool2d(
      GetHpuInput(0), kernel_size, stride, pad, ceil, count_include_pad, {});
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_f32) {
  GenerateInputs(1, {{20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::avg_pool2d(
      GetCpuInput(0), kernel_size, stride, pad, ceil, count_include_pad, {});
  auto result = torch::avg_pool2d(
      GetHpuInput(0), kernel_size, stride, pad, ceil, count_include_pad, {});
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_pad) {
  GenerateInputs(1, {{1, 2, 4, 4}});
  std::vector<int64_t> kernel_size = {3, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {1, 1};
  bool ceil = false;
  bool count_include_pad = true;
  int64_t divisor = 5;

  auto expected = torch::avg_pool2d(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  auto result = torch::avg_pool2d(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_2_2) {
  GenerateInputs(1, {{1, 1, 2, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> kernel_size = {2};
  std::vector<int64_t> stride = {2};
  std::vector<int64_t> pad = {0};
  bool ceil = false;
  bool count_include_pad = false;
  int64_t divisor = 8;

  auto expected = torch::avg_pool2d(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  auto result = torch::avg_pool2d(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_3_3) {
  GenerateInputs(1, {{1, 1, 3, 3}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> pad = {1, 1};
  bool ceil = true;
  bool count_include_pad = true;
  int64_t divisor = 6;

  auto expected = torch::avg_pool2d(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  auto result = torch::avg_pool2d(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_diffpad) {
  GenerateInputs(1, {{20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {4};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {1, 2};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_3d_out_f32) {
  GenerateInputs(1, {{4, 7, 5}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_out_f32) {
  GenerateInputs(1, {{20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_out_pad) {
  GenerateInputs(1, {{1, 2, 4, 4}});
  std::vector<int64_t> kernel_size = {3, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {1, 1};
  bool ceil = false;
  bool count_include_pad = true;
  int64_t divisor = 5;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor,
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor,
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_out_2_2) {
  GenerateInputs(1, {{1, 1, 2, 2}});
  torch::ScalarType dtype = torch::kFloat;
  std::vector<int64_t> kernel_size = {2};
  std::vector<int64_t> stride = {2};
  std::vector<int64_t> pad = {0};
  bool ceil = false;
  bool count_include_pad = false;
  int64_t divisor = 8;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor,
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor,
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_out_3_3) {
  GenerateInputs(1, {{1, 1, 3, 3}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {1, 1};
  std::vector<int64_t> pad = {1, 1};
  bool ceil = true;
  bool count_include_pad = true;
  int64_t divisor = 6;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor,
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor,
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_out_diffpad) {
  GenerateInputs(1, {{20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {4};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {1, 2};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_out_diffstride) {
  GenerateInputs(1, {{20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 3};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      expected);
  torch::avg_pool2d_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_bwd_f32) {
  GenerateInputs(2, {{20, 16, 25, 16}, {20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;
  int64_t divisor = 6;

  auto expected = torch::avg_pool2d_backward(
      GetCpuInput(0),
      GetCpuInput(1),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  auto result = torch::avg_pool2d_backward(
      GetHpuInput(0),
      GetHpuInput(1),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      divisor);
  Compare(expected, result);
}

TEST_F(HpuOpTest, avg_pool2d_bwd_out_f32) {
  GenerateInputs(2, {{20, 16, 25, 16}, {20, 16, 50, 32}});
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> stride = {2, 2};
  std::vector<int64_t> pad = {0, 0};
  bool ceil = false;
  bool count_include_pad = false;

  auto expected = torch::empty(0);
  auto result = torch::empty(0, "hpu");

  torch::avg_pool2d_backward_outf(
      GetCpuInput(0),
      GetCpuInput(1),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      expected);
  torch::avg_pool2d_backward_outf(
      GetHpuInput(0),
      GetHpuInput(1),
      kernel_size,
      stride,
      pad,
      ceil,
      count_include_pad,
      {},
      result);
  Compare(expected, result);
}