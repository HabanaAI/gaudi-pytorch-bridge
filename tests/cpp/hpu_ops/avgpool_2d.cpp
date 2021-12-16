/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};

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