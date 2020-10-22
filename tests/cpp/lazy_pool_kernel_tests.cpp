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

class LazyPoolKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyPoolKernelTest, MaxPoolBWDTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  // fwd propga
  torch::Tensor outHabana1 = torch::max_pool2d(tHabanaX, 2, 1);
  torch::Tensor outHabana = torch::relu(outHabana1);

  // bwd propga with dummy grad tensor
  auto grad_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 3, 3, 3});
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
  outHabana.backward({tHabanaG}, false, true);

  // Match lazy IR graph
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(outHabana));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  auto out_string = IrGraphDumpUtil::ToText(a);

  EXPECT_EQ(
      out_string.find(
          "IR {\n"
          "  %0 = hpu::input()\n"
          "  %1 = aten::maxpool2d_overidable(%0), kernel_size=[2], stride=[1], padding=[0], dilation=[1], transposed=[0]\n"
          "  %2 = aten::relu(%1.0), ROOT=0\n"
          "}"),
      0);

  // auto expected = torch::tensor({11952}, torch::kFloat);
  // EXPECT_EQ(outHabana.sizes(), expected.view({1, 1, 1, 1}).sizes());
  // ASSERT_TRUE(torch::allclose(outHabana.to(torch::kCPU), expected));
  unsetenv("PT_HPU_LAZY_MODE");
}

