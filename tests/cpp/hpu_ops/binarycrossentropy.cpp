/*******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "util.h"

class HpuOpTest : public HpuOpTestUtil {};
/*
NOTE:
1.BCE Bwd variant without weight tensor, when reduction is sum ,testcase fails.
2.BCE Bwd variant when weight tensor is added, testcase fails for all
reduction modes(Mean,Sum and None)
3.Issue raised for Value Mismatch:https://jira.habana-labs.com/browse/SW-70409
*/

TEST_F(HpuOpTest, bce_usual_3D_sum) {
  const std::vector<int64_t> size = {8, 3, 2};
  GenerateInputs(3, {size, size, {8, 3, 1}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::binary_cross_entropy(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum);
  auto result = torch::binary_cross_entropy(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
      at::Reduction::Sum);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_usual_4D_none_bf16) {
  const std::vector<int64_t> size = {4, 8, 3, 2};
  GenerateInputs(3, {size, size, {4, 8, 3, 1}});
  torch::ScalarType dtype = torch::kBFloat16;

  auto expected = torch::binary_cross_entropy(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ {},
      at::Reduction::None);
  auto result = torch::binary_cross_entropy(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ {},
      at::Reduction::None);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_bwd_2D_mean_bf16) {
  torch::ScalarType dtype = torch::kBFloat16;
  auto grad_out = torch::randn({1}, dtype);

  const std::vector<int64_t> size = {2, 4};
  GenerateInputs(2, {size, size}, dtype);

  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);

  auto expected = torch::binary_cross_entropy_backward(
      grad_out.to(torch::kFloat),
      torch::sigmoid(GetCpuInput(0)).to(torch::kFloat),
      /*target*/ GetCpuInput(1).to(torch::kFloat),
      /*weight*/ {},
      at::Reduction::Mean);
  auto result = torch::binary_cross_entropy_backward(
      hgrad_out,
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ GetHpuInput(1),
      /*weight*/ {},
      at::Reduction::Mean);
  Compare(expected.to(dtype), result, 1.5e-01, 1e-01);
}

TEST_F(HpuOpTest, bce_bwd_3D_mean) {
  auto grad_out = torch::randn({1});

  const std::vector<int64_t> size = {3, 4, 5};
  GenerateInputs(2, {size, size});

  torch::ScalarType dtype = torch::kFloat;
  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);

  auto expected = torch::binary_cross_entropy_backward(
      grad_out,
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ GetCpuInput(1),
      /*weight*/ {},
      at::Reduction::Mean);
  auto result = torch::binary_cross_entropy_backward(
      hgrad_out,
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ GetHpuInput(1),
      /*weight*/ {},
      at::Reduction::Mean);

  Compare(expected, result);
}
TEST_F(HpuOpTest, bce_out_3D_sum) {
  const std::vector<int64_t> size = {8, 3, 2};
  GenerateInputs(3, {size, size, {8, 3, 1}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
      at::Reduction::Sum,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_out_2D_none) {
  const std::vector<int64_t> size = {6, 1};
  GenerateInputs(3, {size, size, {1}});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::None,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
      at::Reduction::None,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_out_4D_mean) {
  const std::vector<int64_t> size = {9, 7, 5, 2};
  GenerateInputs(3, {size, size, size});

  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Mean,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid((GetHpuInput(1))),
      /*weight*/ GetHpuInput(2),
      at::Reduction::Mean,
      result);
  Compare(expected, result);
}

/*
BCE out variant - For higher dimension, 5D Input Mismatch Results
Issue raised: https://jira.habana-labs.com/browse/SW-73402
*/
TEST_F(HpuOpTest, bce_out_5D_sum) {
  const std::vector<int64_t> size = {1, 2, 3, 4, 2};
  GenerateInputs(3, {size, size, size});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ GetCpuInput(2),
      at::Reduction::Sum,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
      at::Reduction::Sum,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_out_5D_mean) {
  const std::vector<int64_t> size = {1, 6, 5, 9, 8};
  GenerateInputs(2, {size, size});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ {},
      at::Reduction::Mean,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ {},
      at::Reduction::Mean,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_out_5D_none) {
  const std::vector<int64_t> size = {1, 24, 12, 7, 9};
  GenerateInputs(2, {size, size});
  torch::ScalarType dtype = torch::kFloat;

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ torch::sigmoid(GetCpuInput(1)),
      /*weight*/ {},
      at::Reduction::None,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ {},
      at::Reduction::None,
      result);

  Compare(expected, result);
}

TEST_F(HpuOpTest, bce_bwd_out_2D_mean) {
  auto grad_out = torch::randn({1});

  const std::vector<int64_t> size = {3, 1};
  GenerateInputs(2, {size, size});

  torch::ScalarType dtype = torch::kFloat;
  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);

  auto expected = torch::empty(size, dtype);
  auto result = torch::empty(size, torch::TensorOptions().device("hpu"));

  torch::binary_cross_entropy_backward_outf(
      grad_out,
      torch::sigmoid(GetCpuInput(0)),
      /*target*/ GetCpuInput(1),
      /*weight*/ {},
      at::Reduction::Mean,
      expected);
  torch::binary_cross_entropy_backward_outf(
      hgrad_out,
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ GetHpuInput(1),
      /*weight*/ {},
      at::Reduction::Mean,
      result);

  Compare(expected, result);
}

/*
For the below two testcases : bce_out_3D_none_bf16 and bce_bwd_out_3D_mean_bf16
BFloat16 dtype is not supported in PyTorch, so inputs are generated with bf16
and converted to f32 for cpu. Cpu result is converted back again to bf16 and it
is compared with Hpu. Since default tolerance is not supported, we have tuned
the tolerance values.
*/
TEST_F(HpuOpTest, bce_out_3D_none_bf16) {
  torch::ScalarType dtype = torch::kBFloat16;

  const std::vector<int64_t> size = {5, 2, 7};
  GenerateInputs(3, {size, size, size}, dtype);

  auto expected = torch::empty(size);
  auto result = torch::empty(size).to(dtype).to(torch::kHPU);

  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetCpuInput(0)).to(torch::kFloat),
      /*target*/ torch::sigmoid(GetCpuInput(1)).to(torch::kFloat),
      /*weight*/ GetCpuInput(2).to(torch::kFloat),
      at::Reduction::None,
      expected);
  torch::binary_cross_entropy_outf(
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ torch::sigmoid(GetHpuInput(1)),
      /*weight*/ GetHpuInput(2),
      at::Reduction::None,
      result);

  Compare(expected.to(dtype), result, 1.5e-01, 1e-01);
}

TEST_F(HpuOpTest, bce_bwd_out_3D_mean_bf16) {
  torch::ScalarType dtype = torch::kBFloat16;
  auto grad_out = torch::randn({1}, dtype);

  const std::vector<int64_t> size = {2, 4, 3};
  GenerateInputs(2, {size, size}, dtype);

  grad_out = grad_out.to(dtype);
  auto hgrad_out = grad_out.to(torch::kHPU, dtype);

  auto expected = torch::empty(size);
  auto result = torch::empty(size).to(dtype).to(torch::kHPU);

  torch::binary_cross_entropy_backward_outf(
      grad_out.to(torch::kFloat),
      torch::sigmoid(GetCpuInput(0)).to(torch::kFloat),
      /*target*/ GetCpuInput(1).to(torch::kFloat),
      /*weight*/ {},
      at::Reduction::Mean,
      expected);
  torch::binary_cross_entropy_backward_outf(
      hgrad_out,
      torch::sigmoid(GetHpuInput(0)),
      /*target*/ GetHpuInput(1),
      /*weight*/ {},
      at::Reduction::Mean,
      result);

  Compare(expected.to(dtype), result, 1.5e-01, 1e-01);
}
