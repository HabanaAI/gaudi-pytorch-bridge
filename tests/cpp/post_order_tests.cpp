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

#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace torch;

class PostOrderTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    ForceMode(1);
    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }
};

TEST_F(PostOrderTest, poTestAdd) {
  // test case for result = add(tensor1, tensor2, alpha)
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHPU);
  Scalar alpha = 1.0;
  auto result = add_tensor_hpu_lazy(tensor_in1, tensor_in2, alpha);
  auto hl_result = SyncAndGetHbLazyTensor(result);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  bool cond = (str.find("prim::constant") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[1]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[2]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[3]->ToString();
  cond = (str.find("hpu::add") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 2);
}

TEST_F(PostOrderTest, poTestFill) {
  // test case for result.fill_(val)
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  Scalar alpha = 1.0;

  tensor_in1.fill_(alpha);
  auto hl_result = SyncAndGetHbLazyTensor(tensor_in1);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto str = po_data.post_order[0]->ToString();
  EXPECT_TRUE(str.find("prim::constant") != string::npos);

  str = po_data.post_order[1]->ToString();
  EXPECT_TRUE(str.find("hpu::input") != string::npos);

  str = po_data.post_order[3]->ToString();
  EXPECT_TRUE(str.find("aten::fill_") != string::npos);

  EXPECT_TRUE(po_data.inputs.size() == 1);
  EXPECT_TRUE(po_data.outputs.size() == 1);

  std::vector<at::Tensor> input_list{tensor_in1};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("= prim::Constant[value=1.]", 1)
      ->check("= aten::fill_")
      ->run(*hlexec->get_graph());
}

TEST_F(PostOrderTest, poTestCommonInput) {
  // test case for
  // t = add(tensor1, tensor2, alpha)
  // result = add(t, tensor2, beta)
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHPU);
  Scalar alpha = 1.0f, beta = 2.0f;
  auto result = add_tensor_hpu_lazy(tensor_in1, tensor_in2, alpha);

  auto result2 = add_tensor_hpu_lazy(result, tensor_in2, beta);
  auto hl_result = SyncAndGetHbLazyTensor(result2);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  bool cond = (str.find("prim::constant") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[1]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[2]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[3]->ToString();
  cond = (str.find("aten::mul") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[4]->ToString();
  cond = (str.find("prim::constant") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[5]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[6]->ToString();
  cond = (str.find("hpu::add") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[7]->ToString();
  cond = (str.find("hpu::add") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 3);
}

TEST_F(PostOrderTest, poTestAddInplace) {
  // test case for tensor1 = add(tensor1, tensor2, alpha)
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHPU);
  tensor_in1 = tensor_in1.add_(tensor_in2);
  auto hl_result = SyncAndGetHbLazyTensor(tensor_in1);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  bool cond = (str.find("prim::constant") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[1]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[2]->ToString();
  cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[3]->ToString();
  cond = (str.find("hpu::add") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 2);
}

TEST_F(PostOrderTest, poTestReluInplace) {
  // test case for tensor1 = relu(tensor1)
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  tensor_in1 = tensor_in1.relu_();
  auto hl_result = SyncAndGetHbLazyTensor(tensor_in1);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  auto cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[1]->ToString();
  cond = (str.find("aten::relu_") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 1);
}

TEST_F(PostOrderTest, TestInplaceAndD2H_case1) {
  torch::Tensor c0 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randn({20, 5}, torch::requires_grad(false));

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);

  c0.add_(c1);
  torch::Tensor c2 = torch::abs(c0);
  torch::Tensor c3 = torch::abs(c0);

  h0.add_(h1);
  torch::Tensor h2 = torch::abs(h0);
  torch::Tensor h3 = torch::abs(h0);

  torch::Tensor h0_c = h0.to(torch::kCPU);
  torch::Tensor h2_c = h2.to(torch::kCPU);
  torch::Tensor h3_c = h3.to(torch::kCPU);

  EXPECT_TRUE(allclose(c0, h0_c));
  EXPECT_TRUE(allclose(c2, h2_c));
  EXPECT_TRUE(allclose(c3, h3_c));
}

TEST_F(PostOrderTest, TestInplaceAndD2H_case2) {
  torch::Tensor c0 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randn({20, 5}, torch::requires_grad(false));

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);

  c0.add_(c1);
  c0.add_(c1);
  torch::Tensor c2 = torch::abs(c0);
  torch::Tensor c3 = torch::abs(c0);
  c0.add_(c1);
  c0.add_(c1);

  h0.add_(h1);
  h0.add_(h1);
  torch::Tensor h2 = torch::abs(h0);
  torch::Tensor h3 = torch::abs(h0);
  h0.add_(h1);
  h0.add_(h1);
  HbLazyTensor::StepMarker({});

  torch::Tensor h0_c = h0.to(torch::kCPU);
  torch::Tensor h2_c = h2.to(torch::kCPU);
  torch::Tensor h3_c = h3.to(torch::kCPU);

  EXPECT_TRUE(allclose(c0, h0_c));
  EXPECT_TRUE(allclose(c2, h2_c));
  EXPECT_TRUE(allclose(c3, h3_c));
}

TEST_F(PostOrderTest, TestInplaceAndD2H_case3) {
  torch::Tensor c0 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randn({20, 5}, torch::requires_grad(false));

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);

  c0.add_(c1);
  c0.add_(c1);
  c0.add_(c1);

  h0.add_(h1);

  h0.add_(h1);
  h0.add_(h1);

  HbLazyTensor::StepMarker({});
  torch::Tensor h0_c = h0.to(torch::kCPU);

  EXPECT_TRUE(allclose(c0, h0_c));
}

TEST_F(PostOrderTest, TestInplaceAndD2H_case4) {
  torch::Tensor c0 = torch::randint(5, {1, 2}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randint(5, {1, 2}, torch::requires_grad(false));

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);

  c0.add_(c1);

  c0.add_(c1);
  c0.add_(c1);

  h0.add_(h1);
  torch::Tensor h0_t = h0.to(torch::kCPU);

  h0.add_(h1);
  h0.add_(h1);

  torch::Tensor h0_c = h0.to(torch::kCPU);
  EXPECT_TRUE(allclose(c0, h0_c));
}

TEST_F(PostOrderTest, TestInplaceAndD2H_case5) {
  torch::Tensor c0 = torch::randint(5, {1, 2}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randint(5, {1, 2}, torch::requires_grad(false));

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);

  c0.add_(c1);

  torch::Tensor c2 = torch::abs(c0);
  torch::Tensor c3 = torch::abs(c0);
  c0.add_(c1);

  h0.add_(h1);
  torch::Tensor h2 = torch::abs(h0);
  torch::Tensor h3 = torch::abs(h0);

  torch::Tensor h0_t = h0.to(torch::kCPU);

  h0.add_(h1);
  HbLazyTensor::StepMarker({});

  torch::Tensor h0_c = h0.to(torch::kCPU);
  torch::Tensor h2_c = h2.to(torch::kCPU);

  torch::Tensor h3_c = h3.to(torch::kCPU);

  EXPECT_TRUE(allclose(c0, h0_c));
  EXPECT_TRUE(allclose(c2, h2_c));
  EXPECT_TRUE(allclose(c3, h3_c));
}

TEST_F(PostOrderTest, TestInplaceAndD2H_case6) {
  torch::Tensor c0 = torch::randint(5, {1, 2}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randint(5, {1, 2}, torch::requires_grad(false));

  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);

  c0.add_(c1);

  c0.add_(c1);
  c0.add_(c1);
  torch::Tensor c2 = torch::abs(c0);
  torch::Tensor c3 = torch::abs(c0);

  h0.add_(h1);
  torch::Tensor h0_t = h0.to(torch::kCPU);

  h0.add_(h1);
  h0.add_(h1);

  HbLazyTensor::StepMarker({});
  torch::Tensor h0_c = h0.to(torch::kCPU);
  torch::Tensor h2 = torch::abs(h0);
  torch::Tensor h3 = torch::abs(h0);
  h0_c = h0.to(torch::kCPU);
  torch::Tensor h2_c = h2.to(torch::kCPU);

  h0_c = h0.to(torch::kCPU);
  torch::Tensor h3_c = h3.to(torch::kCPU);
  EXPECT_TRUE(allclose(c0, h0_c));
  EXPECT_TRUE(allclose(c2, h2_c));
  EXPECT_TRUE(allclose(c3, h3_c));
}

TEST_F(PostOrderTest, D2H_Test) {
  torch::Tensor c0 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor c1 = torch::randn({20, 5}, torch::requires_grad(false));
  torch::Tensor h0 = c0.to(torch::kHPU);
  torch::Tensor h1 = c1.to(torch::kHPU);
  auto c2 = c0.add(c1);
  auto c3 = c0.add(c1);

  auto c4 = torch::relu(c2);
  auto c5 = torch::relu(c3);
  auto c6 = torch::relu(c4);

  auto h2 = h0.add(h1);
  auto h3 = h0.add(h1);

  auto h4 = torch::relu(h2);
  auto h5 = torch::relu(h3);
  auto h6 = torch::relu(h4);

  torch::Tensor h4_c = h4.to(torch::kCPU);
  torch::Tensor h5_c = h5.to(torch::kCPU);
  torch::Tensor h6_c = h6.to(torch::kCPU);

  EXPECT_TRUE(allclose(c4, h4_c));
  EXPECT_TRUE(allclose(c5, h5_c));
  EXPECT_TRUE(allclose(c5, h6_c));
}

TEST_F(PostOrderTest, poTestCat) {
  // test case for result = add(tensor1, tensor2, alpha)
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHPU);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHPU);

  auto result = torch::cat({torch::neg(tensor_in1), tensor_in2}, 0);
  auto hl_result = SyncAndGetHbLazyTensor(result);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  EXPECT_TRUE(po_data.outputs.size() == 1);
}
