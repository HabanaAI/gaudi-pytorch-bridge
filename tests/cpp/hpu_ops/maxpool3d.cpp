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

TEST_F(HpuOpTest, maxpool_3d_with_indices_4D_input) {
  GenerateInputs(1, {{2, 3, 6, 6}});
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = true;

  auto expected = torch::max_pool3d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto result = torch::max_pool3d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  // max_pool3d_with_indices will return 2 outputs Indices tensor and
  // output tensor. But here we are comparing only output tensor because in
  // pytorch the returend indices tensor contains  indices relative to input
  // feature map but indices tensor from TPC contains indices relative to
  // kernel window. Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  Compare(std::get<0>(expected), std::get<0>(result));
}

TEST_F(HpuOpTest, maxpool_3d_with_indices_5D_input) {
  GenerateInputs(1, {{1, 1, 3, 7, 8}});
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = false;

  auto expected = torch::max_pool3d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto result = torch::max_pool3d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  // max_pool3d_with_indices will return 2 outputs Indices tensor and
  // output tensor. But here we are comparing only output tensor because in
  // pytorch the returend indices tensor contains  indices relative to input
  // feature map but indices tensor from TPC contains indices relative to
  // kernel window. Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  Compare(std::get<0>(expected), std::get<0>(result));
}

TEST_F(HpuOpTest, maxpool_3d_with_indices_f32_4D_input) {
  GenerateInputs(1, {{1, 3, 7, 8}}, torch::kFloat);
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = true;

  auto expected = torch::max_pool3d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto result = torch::max_pool3d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  Compare(std::get<0>(expected), std::get<0>(result));
}

TEST_F(HpuOpTest, maxpool_3d_with_indices_f32_5D_input) {
  GenerateInputs(1, {{1, 1, 3, 7, 8}}, torch::kFloat);
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = true;

  auto expected = torch::max_pool3d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto result = torch::max_pool3d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  Compare(std::get<0>(expected), std::get<0>(result));
}

TEST_F(HpuOpTest, maxpool_3d_with_indices_backward_5D_input) {
  GenerateInputs(1, {{1, 2, 3, 6, 6}});
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = true;

  // max_pool3d with indces will return 2 outputs Indices tensor and
  // output tensor. In pytorch the returend indices tensor contains  indices
  // relative to input feature map but indices tensor from TPC contains indices
  // relative to kernel window. So for testing backward first we will execute
  // the forward operator and then that output will be passed to backward
  // operator and will compare the backward result.
  // Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  auto expected = torch::max_pool3d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto result = torch::max_pool3d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  // Backward
  auto expected_bwd = torch::max_pool3d_with_indices_backward(
      std::get<0>(expected),
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      std::get<1>(expected));
  auto result_bwd = torch::max_pool3d_with_indices_backward(
      std::get<0>(result),
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      std::get<1>(result));
  Compare(expected_bwd, result_bwd);
}

TEST_F(HpuOpTest, maxpool_3d_with_indices_backward_4D_input) {
  GenerateInputs(1, {{2, 3, 6, 6}});
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = true;

  // max_pool3d with indces will return 2 outputs Indices tensor and
  // output tensor. In pytorch the returend indices tensor contains  indices
  // relative to input feature map but indices tensor from TPC contains indices
  // relative to kernel window. So for testing backward first we will execute
  // the forward operator and then that output will be passed to backward
  // operator and will compare the backward result.
  // Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  auto expected = torch::max_pool3d_with_indices(
      GetCpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);
  auto result = torch::max_pool3d_with_indices(
      GetHpuInput(0), kernel_size, stride, pad_size, dilation, ceil_mode);

  // Backward
  auto expected_bwd = torch::max_pool3d_with_indices_backward(
      std::get<0>(expected),
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      std::get<1>(expected));
  auto result_bwd = torch::max_pool3d_with_indices_backward(
      std::get<0>(result),
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      std::get<1>(result));
  Compare(expected_bwd, result_bwd);
}
// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263), so that the implementation is
// commented till the issue got resolved.
/**TEST_F(HpuOpTest, maxpool_3d_with_indices_out) {
  GenerateInputs(1, {{1, 2, 6, 9, 9}});
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
  bool ceil_mode = true;

  torch::ScalarType dtype = torch::kFloat;
  auto out = torch::empty(0, dtype);
  auto hout = torch::empty(0, torch::TensorOptions(dtype).device("hpu"));

  auto indices = torch::empty(0, torch::kInt64);
  auto hindices =
      torch::empty(0, torch::TensorOptions(torch::kInt64).device("hpu"));

  torch::max_pool3d_with_indices_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      out,
      indices);
  torch::max_pool3d_with_indices_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      hout,
      hindices);

  // max_pool3d_with_indices_outf will return 2 outputs Indices tensor and
  // output tensor. But here we are comparing only output tensor because in
  // pytorch the returend indices tensor contains  indices relative to input
  // feature map but indices tensor from TPC contains indices relative to kernel
  // window.
  // Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882

  Compare(out, hout);
}**/

// Since the out varriant intices tensor has some issue
// (https://jira.habana-labs.com/browse/SW-74263), so that the implementation is
// commented till the issue got resolved.
/**TEST_F(HpuOpTest, maxpool_3d_with_indices_out_backward) {
  GenerateInputs(1, {{1, 2, 6, 9, 9}});
  std::vector<int64_t> kernel_size = {{3, 3, 3}};
  std::vector<int64_t> stride = {{3, 3, 3}};
  std::vector<int64_t> pad_size = {{1, 1, 1}};
  std::vector<int64_t> dilation = {{1, 1, 1}};
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

  // max_pool3d with indces will return 2 outputs Indices tensor and
  // output tensor. In pytorch the returend indices tensor contains  indices
  // relative to input feature map but indices tensor from TPC contains indices
  // relative to kernel window. So for testing backward first we will execute
  // the forward operator and then that output will be passed to backward
  // operator and will compare the backward result.
  // Jia rased for the above issue -
  // https://jira.habana-labs.com/browse/SW-73882
  torch::max_pool3d_with_indices_outf(
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      expected_tensor,
      expected_indices);
  torch::max_pool3d_with_indices_outf(
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      result_tensor,
      result_indices);

  // Backward

  auto expected_bwd = torch::empty(0, dtypef);
  auto result_bwd = torch::empty(0, torch::TensorOptions(dtypef).device("hpu"));

  torch::max_pool3d_with_indices_backward_outf(
      expected_tensor,
      GetCpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      expected_indices,
      expected_bwd);

  // Since there is a error in the indices dtype of out fwd variant the correct
  // indices will be generated in HPU Float type. But for bwd variant the
  // expected dtype of idices is Byte ot Int16. So we are forced to add a
  // convertion as done below.Then only the test case is passed.
  // https://jira.habana-labs.com/browse/SW-74263
  torch::max_pool3d_with_indices_backward_outf(
      result_tensor,
      GetHpuInput(0),
      kernel_size,
      stride,
      pad_size,
      dilation,
      ceil_mode,
      result_indices,
      result_bwd);
  Compare(expected_bwd, result_bwd);
}**/