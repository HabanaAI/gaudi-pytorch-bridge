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

TEST_F(HpuOpTest, reflection_pad1d) {
  GenerateInputs(1, {{3, 3}});
  // tpc expects the pad array to have the values in the order pad_before
  // for each dim followed by pad_after for each dim
  std::vector<int64_t> pad_size = {{2, 1}};
  auto expected = torch::reflection_pad1d(GetCpuInput(0), pad_size);
  auto result = torch::reflection_pad1d(GetHpuInput(0), pad_size);
  Compare(expected, result);
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