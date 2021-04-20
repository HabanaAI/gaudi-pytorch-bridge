#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace torch;

TEST(PostOrderTest, poTestAdd) {
  // test case for result = add(tensor1, tensor2, alpha)
  setenv("PT_HPU_LAZY_MODE", "1", 0);
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHABANA);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHABANA);
  Scalar alpha = 1.0;
  auto result = add_tensor_hpu_lazy(tensor_in1, tensor_in2, alpha);
  auto hl_result = GetHbLazyTensor(result);

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
  cond = (str.find("aten::add") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 2);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(PostOrderTest, poTestFill) {
  // test case for result.fill_(val)
  setenv("PT_HPU_LAZY_MODE", "1", 0);
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHABANA);
  Scalar alpha = 1.0;

  tensor_in1.fill_(alpha);
  auto hl_result = GetHbLazyTensor(tensor_in1);

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
      .check_count("prim::Constant[value=1.]", 1)
      ->check("aten::fill_")
      ->run(*hlexec->get_graph());

  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(PostOrderTest, poTestCommonInput) {
  // test case for
  // t = add(tensor1, tensor2, alpha)
  // result = add(t, tensor2, beta)
  setenv("PT_HPU_LAZY_MODE", "1", 0);
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHABANA);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHABANA);
  Scalar alpha = 1.0f, beta = 2.0f;
  auto result = add_tensor_hpu_lazy(tensor_in1, tensor_in2, alpha);

  auto result2 = add_tensor_hpu_lazy(result, tensor_in2, beta);
  auto hl_result = GetHbLazyTensor(result2);

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
  cond = (str.find("aten::add") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[7]->ToString();
  cond = (str.find("aten::add") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 3);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(PostOrderTest, poTestAddInplace) {
  // test case for tensor1 = add(tensor1, tensor2, alpha)
  setenv("PT_HPU_LAZY_MODE", "1", 0);
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHABANA);
  torch::Tensor tensor_in2 = torch::randn({2, 3}).to(torch::kHABANA);
  tensor_in1 = tensor_in1.add_(tensor_in2);
  auto hl_result = GetHbLazyTensor(tensor_in1);

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
  str = po_data.post_order[4]->ToString();
  cond = (str.find("aten::add") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 2);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(PostOrderTest, poTestReluInplace) {
  // test case for tensor1 = relu(tensor1)
  setenv("PT_HPU_LAZY_MODE", "1", 0);
  torch::Tensor tensor_in1 = torch::randn({2, 3}).to(torch::kHABANA);
  tensor_in1 = tensor_in1.relu_();
  auto hl_result = GetHbLazyTensor(tensor_in1);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);
  auto str = po_data.post_order[0]->ToString();
  auto cond = (str.find("hpu::input") != string::npos);
  EXPECT_TRUE(cond);
  str = po_data.post_order[2]->ToString();
  cond = (str.find("aten::relu") != string::npos);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.outputs.size() == 1);
  EXPECT_TRUE(cond);
  EXPECT_TRUE(po_data.inputs.size() == 1);
  unsetenv("PT_HPU_LAZY_MODE");
}
