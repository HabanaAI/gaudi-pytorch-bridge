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

TEST_F(HpuOpTest, maxpool_2d_with_indices) {
  GenerateInputs(1, {{1, 2, 7, 9}});
  std::vector<int64_t> kernel_size = {{3, 3}};
  std::vector<int64_t> stride = {{3, 3}};
  std::vector<int64_t> pad_size = {{1, 1}};
  std::vector<int64_t> dilation = {{1, 1}};
  bool ceil_mode = false;
  auto cpu_out = torch::max_pool2d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto hpu_out = torch::max_pool2d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  Compare(std::get<0>(cpu_out), std::get<0>(hpu_out));
}

TEST_F(HpuOpTest, maxpool_2d_with_indices_backward) {
  GenerateInputs(1, {{1, 2, 6, 6}});
  std::vector<int64_t> kernel_size = {{3, 3}};
  std::vector<int64_t> stride = {{3, 3}};
  std::vector<int64_t> pad_size = {{1, 1}};
  std::vector<int64_t> dilation = {{1, 1}};
  bool ceil_mode = true;

  auto maxpool_cpu = torch::max_pool2d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto maxpool_hpu = torch::max_pool2d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  auto expected_tensor = std::get<0>(maxpool_cpu);
  auto expected_indices = std::get<1>(maxpool_cpu);
  auto result_tensor = std::get<0>(maxpool_hpu);
  auto result_indices = std::get<1>(maxpool_hpu);

  auto expected_gradinp = torch::max_pool2d_with_indices_backward(
      expected_tensor,
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      expected_indices);

  auto result_gradinp = torch::max_pool2d_with_indices_backward(
      result_tensor,
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      result_indices);
  Compare(expected_gradinp, result_gradinp);
}

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263), so that the implementation is
// commented till the issue got resolved.
/**TEST_F(HpuOpTest, maxpool_2d_with_indices_out) {
  GenerateInputs(1, {{1, 2, 7, 9}});
  std::vector<int64_t> kernel_size = {{3, 3}};
  std::vector<int64_t> stride = {{3, 3}};
  std::vector<int64_t> pad_size = {{1, 1}};
  std::vector<int64_t> dilation = {{1, 1}};
  bool ceil_mode = false;
  torch::ScalarType dtype = torch::kInt64;
  torch::ScalarType dtypef = torch::kFloat;
  auto expected_tensor = torch::empty(0, dtypef);
  auto expected_indices = torch::empty(0, dtype);
  auto result_tensor =
      torch::empty(0, torch::TensorOptions(dtypef).device("hpu"));
  auto result_indices =
      torch::empty(0, torch::TensorOptions(dtype).device("hpu"));
  torch::max_pool2d_with_indices_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      expected_tensor,
      expected_indices);
  torch::max_pool2d_with_indices_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      result_tensor,
      result_indices);

  // max_pool2d_with_indices will return 2 outputs Indices tensor and
  // output tensor. But here we are comparing only output tensor because in
  // pytorch the returend indices tensor contains  indices relative to input
  // feature map but indices tensor from TPC contains indices relative to kernel
  // window.
  // Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  Compare(expected_tensor, result_tensor);
}

TEST_F(HpuOpTest, maxpool_2d_with_indices_out_backward) {
  GenerateInputs(1, {{2, 6, 6}});
  std::vector<int64_t> kernel_size = {{3, 3}};
  std::vector<int64_t> stride = {{3, 3}};
  std::vector<int64_t> pad_size = {{1, 1}};
  std::vector<int64_t> dilation = {{1, 1}};
  bool ceil_mode = true;
  torch::ScalarType dtype = torch::kInt64;
  torch::ScalarType dtypef = torch::kFloat;
  auto expected_tensor = torch::empty(0, dtypef);
  auto expected_indices = torch::empty(0, dtype);
  auto result_tensor =
      torch::empty(0, torch::TensorOptions(dtypef).device("hpu"));

  // If we use Indices dtype for HPU as Int or Byte for fwd out variant, then we
  // are getting some junk values in the indices tensor and if we increase the
  // test case dimensions like channel greater than 1 then the test case will
  // failed with mismatch error. But, if we change the indices dtype  of HPU to
  // Float then what ever the test case dimension correct indices will be
  // generated.
  // https://jira.habana-labs.com/browse/SW-74263
  auto result_indices =
      torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  // max_pool2d with indces will return 2 outputs Indices tensor and
  // output tensor. In pytorch the returend indices tensor contains  indices
  // relative to input feature map but indices tensor from TPC contains indices
  // relative to kernel window. So for testing backward first we will execute
  // the forward operator and then that output will be passed to backward
  // operator and will compare the backward result.
  // Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  torch::max_pool2d_with_indices_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      expected_tensor,
      expected_indices);
  torch::max_pool2d_with_indices_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      result_tensor,
      result_indices);

  // Backward
  auto expected_gradinp = torch::empty(0, dtypef);
  auto result_gradinp =
      torch::empty(0, torch::TensorOptions(dtypef).device("hpu"));

  torch::max_pool2d_with_indices_backward_outf(
      expected_tensor,
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      expected_indices,
      expected_gradinp);

  // Since there is a error in the indices dtype of out fwd variant the correct
  // indices will be generated in HPU Float type. But for bwd variant the
  // expected dtype of idices is Byte ot Int16. So we are forced to add a
  // convertion as done below.Then only the test case is passed.
  // https://jira.habana-labs.com/browse/SW-74263
  torch::max_pool2d_with_indices_backward_outf(
      result_tensor,
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      result_indices,
      result_gradinp);
  Compare(expected_gradinp, result_gradinp);
}**/