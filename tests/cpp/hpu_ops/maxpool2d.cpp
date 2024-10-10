/******************************************************************************
 * Copyright (C) 2021-2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "../utils/dtype_supported_on_device.h"
#include "util.h"

class HpuOpTest : public HpuOpTestUtil {
 public:
  void RetainTensorTypeTest(torch::ScalarType dtype) {
    if (!IsDtypeSupportedOnCurrentDevice(dtype)) {
      GTEST_SKIP();
    }
    GenerateInputs(1, {{1, 2, 7, 9}}, {dtype});
    std::vector<int64_t> kernel_size = {{3, 3}};
    std::vector<int64_t> stride = {{3, 3}};
    std::vector<int64_t> pad_size = {{1, 1}};
    std::vector<int64_t> dilation = {{1, 1}};
    bool ceil_mode = false;
    auto hpu_out = torch::max_pool2d_with_indices(
        GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
    auto expectedRetainTEnsorType = torch::kLong;
    EXPECT_EQ(std::get<1>(hpu_out).scalar_type(), expectedRetainTEnsorType);
  }
};

TEST_F(HpuOpTest, maxpool_2d_with_indices_half) {
  RetainTensorTypeTest(torch::kHalf);
}

TEST_F(HpuOpTest, maxpool_2d_with_indices_bfloat16) {
  RetainTensorTypeTest(torch::kBFloat16);
}

TEST_F(HpuOpTest, maxpool_2d_with_indices_float) {
  RetainTensorTypeTest(torch::kFloat32);
}

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

TEST_F(HpuOpTest, maxpool_2d_with_indices_out) {
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

  Compare(expected_tensor, result_tensor);
}

TEST_F(HpuOpTest, maxpool_2d_with_indices_backward_out) {
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

  // Backward
  torch::ScalarType dtypef = torch::kFloat;
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
}