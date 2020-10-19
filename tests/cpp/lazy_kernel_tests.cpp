#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

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
  auto input_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 3, 3}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  auto weight_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 3, 1}); // hwck
  torch::Tensor tHabanaW = weight_tensor.to(torch::kHABANA);

  auto bias_tensor =
      torch::arange(1, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 1, 1, 1});
  torch::Tensor tHabanaB = bias_tensor.to(torch::kHABANA);

  torch::Tensor outConv = torch::conv2d(tHabanaX, tHabanaW, {}, 1, 0, 1, 1);
  torch::Tensor outHabana = torch::relu(outConv);

  // Match lazy IR graph
  auto hl_result = GetHbLazyTensor(outHabana);
  auto ir_value = hl_result.CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("prim::Constant()")
      ->check_count("prim::Constant[value=[1, 1]]", 2)
      ->check("prim::Constant[value=0]")
      ->check("prim::Constant[value=1]")
      ->check("aten::convolution_overidable")
      ->run(*exec.get_graph());

  torch::jit::testing::FileCheck()
      .check_count("prim::Constant[value=[0, 0]]", 2)
      ->run(*exec.get_graph());

  auto out_string = IrGraphDumpUtil::ToText(a);
  EXPECT_EQ(
      out_string.find("IR {\n"
                      "  %0 = hpu::input()\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = aten::convolution_overidable(%1, %0), stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, output_padding=[0, 0], groups=1\n"
                      "  %3 = aten::relu(%2), ROOT=0\n"
                      "}"),
      0);

  // Match expectd output Size&Data
  auto expected = torch::tensor({5265}, torch::kFloat);
  EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  // ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), expected));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyKernelTest, MmMulTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto x = torch::randn({2, 3});
  auto y = torch::randn({3, 3});
  auto z = torch::randn({2, 3});
  torch::Tensor hx = x.to(torch::kHABANA);
  torch::Tensor hy = y.to(torch::kHABANA);
  torch::Tensor hz = z.to(torch::kHABANA);

  auto hy_exp = torch::mm(hx, hy);
  auto hz_exp = torch::mul(hy_exp, hz);
  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(hz_exp));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find("IR {\n"
                      "  %0 = hpu::input()\n"
                      "  %1 = hpu::input()\n"
                      "  %2 = hpu::input()\n"
                      "  %3 = aten::mm(%2, %1)\n"
                      "  %4 = aten::mul(%3, %0), ROOT=0\n"
                      "}"),
      0);

  // Match expectd output
  // ASSERT_TRUE(torch::allclose(hz_exp, hz_exp));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyKernelTest, CatTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor out = torch::cat({hA, hB, hC});

  auto hl_result = GetHbLazyTensor(out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check("Tensor[] = prim::ListConstruct")
      ->check("int = prim::Constant[value=0]")
      ->check("Tensor = aten::cat")
      ->run(*exec.get_graph());
  unsetenv("PT_HPU_LAZY_MODE");
}