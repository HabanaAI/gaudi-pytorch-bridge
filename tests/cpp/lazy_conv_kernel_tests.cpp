#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;

class LazyConvKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyConvKernelTest, ConvReluTest) {
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
      out_string.find(
          "IR {\n"
          "  %0 = hpu::input()\n"
          "  %1 = hpu::input()\n"
          "  %2 = aten::convolution_overidable(%1, %0), stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, output_padding=[0, 0], groups=1\n"
          "  %3 = aten::relu(%2), ROOT=0\n"
          "}"),
      0);

  // Match expectd output Size&Data
  auto expected = torch::tensor({5265}, torch::kFloat);
  EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyConvKernelTest, ConvMaxPoolTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
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
  torch::Tensor outHabana = torch::max_pool2d(outConv, 2, 1);

  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(outHabana));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find(
          "IR {\n"
          "  %0 = hpu::input()\n"
          "  %1 = hpu::input()\n"
          "  %2 = aten::convolution_overidable(%1, %0), stride=[1, 1], padding=[0, 0], dilation=[1, 1], transposed=False, output_padding=[0, 0], groups=1\n"
          "  %3 = aten::maxpool2d_overidable(%2), kernel_size=[2], stride=[1], padding=[0], dilation=[1], transposed=[0], ROOT=0\n"
          "}"),
      0);

  // Match expectd output Size&Data
  auto expected = torch::tensor({11952}, torch::kFloat);
  EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  // ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), expected));
  unsetenv("PT_HPU_LAZY_MODE");
}

