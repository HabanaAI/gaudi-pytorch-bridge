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
  auto expected = torch::empty((2, 3, 7), dtype);
  auto result =
      torch::empty((2, 3, 7), torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad1d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad1d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad2d_input4d_out) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty((2, 3, 13, 10), dtype);
  auto result =
      torch::empty((2, 3, 13, 10), torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad2d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad2d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}

TEST_F(HpuOpTest, replication_pad3d_input5d_out) {
  GenerateInputs(1, {{2, 3, 4, 5, 6}});
  std::vector<int64_t> pad_size = {{2, 3, 4, 5, 6, 7}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty((2, 3, 17, 14, 11), dtype);
  auto result = torch::empty(
      (2, 3, 17, 14, 11), torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad3d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad3d_outf(GetHpuInput(0), pad_size, result);
}

TEST_F(HpuOpTest, replication_pad2d_input4d_out_zero_pad) {
  GenerateInputs(1, {{2, 3, 4, 5}});
  std::vector<int64_t> pad_size = {{0, 0, 0, 0}};
  torch::ScalarType dtype = torch::kFloat;
  auto expected = torch::empty((2, 3, 4, 5), dtype);
  auto result =
      torch::empty((2, 3, 4, 5), torch::TensorOptions(dtype).device("hpu"));
  torch::replication_pad2d_outf(GetCpuInput(0), pad_size, expected);
  torch::replication_pad2d_outf(GetHpuInput(0), pad_size, result);
  Compare(expected, result);
}
