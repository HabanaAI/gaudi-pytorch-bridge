#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>

#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hpu_lazy_tensors.h"

using namespace habana_lazy;

TEST(LazyKernelTest, LazyDoATest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor I = torch::add(hA, hB);
  torch::Tensor out = torch::add(hC, I);
  EXPECT_EQ(out.dim(), 2);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyKernelTest, BasicCopyTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  bool equal = hA_cpu.allclose(A, 0, 0);
  EXPECT_EQ(equal, true);
  unsetenv("PT_HPU_LAZY_MODE");
}
TEST(LazyKernelTest, ConvReluTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor = torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
                          .reshape({1, 3, 3, 3}); //nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  auto weight_tensor = torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
                          .reshape({3, 3, 3, 1}); // hwck
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHABANA);

  auto bias_tensor = torch::arange(1, torch::dtype(torch::kFloat).requires_grad(false))
                          .reshape({1, 1, 1, 1});
  torch::Tensor tHabanaB = bias_tensor.to(torch::kHABANA);

  torch::Tensor outConv = torch::conv2d(tHabanaX, tHabanaW, {}, 1, 0, 1, 1);
  torch::Tensor outHabana = torch::relu(outConv);

  //Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(outHabana));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);
  EXPECT_EQ(
      out_string.find("IR {\n"
                      "  %0 = hpu::input()\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = aten::convolution(%1, %0)\n"
                      "  %3 = aten::relu(%2), ROOT=0\n"
                      "}"),
      0);

  // Match expectd output Size&Data
  auto expected = torch::tensor({5265}, torch::kFloat);
  EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  //ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), expected));
  unsetenv("PT_HPU_LAZY_MODE");
}