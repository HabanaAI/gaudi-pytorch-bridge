#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"
#include "habana_kernels/eager_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"

using namespace habana_lazy;
using namespace torch;

TEST(PostOrderTest, poTest1) {
  // test case for result = add(tensor1, tensor2, alpha)
  setenv("PT_HPU_LAZY_MODE", "1", 1);
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