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
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyConvKernelTest, ConvReluTest) {
  auto input_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 3, 3}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);

  auto weight_tensor =
      torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({3, 3, 3, 1}); // hwck

  auto wt_hwck = weight_tensor.permute({2, 3, 1, 0}).contiguous();
  torch::Tensor tHabanaW = wt_hwck.to(torch::kHABANA);

  torch::Tensor outConv = torch::conv2d(tHabanaX, tHabanaW, {}, 1, 0, 1, 1);
  torch::Tensor outhpu = torch::relu(outConv);
  torch::Tensor out = outhpu.to(torch::kCPU);

  torch::Tensor outConv1 =
      torch::conv2d(input_tensor, weight_tensor, {}, 1, 0, 1, 1);
  torch::Tensor outcpu = torch::relu(outConv1);

  EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
}

TEST_F(LazyConvKernelTest, MaxPool2DTest) {
  auto input_tensor = torch::randn({20, 16, 50, 32});

  torch::Tensor outHabana = torch::max_pool2d(input_tensor, 2, 1);
  auto out = outHabana.to(torch::kCPU);

  torch::Tensor outcpu = torch::max_pool2d(input_tensor, 2, 1);

  EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
}

TEST_F(LazyConvKernelTest, ConvolutionBackward) {
  auto grad_output = torch::randn({2, 6, 2, 3}, torch::requires_grad(false));
  auto input = torch::randn({2, 5, 3, 4}, torch::requires_grad(false));
  auto weight = torch::randn({2, 2, 5, 6}, torch::requires_grad(false));

  auto h_grad_output = grad_output.to(torch::kHABANA);
  auto hinput = input.to(torch::kHABANA);
  auto hweight = weight.to(torch::kHABANA);

  torch::Tensor out1, out2, out3;
  std::tie(out1, out2, out3) = convolution_backward_overrideable(
      h_grad_output,
      hinput,
      hweight,
      {1, 1},
      {0, 0},
      {1, 1},
      false,
      {0, 0},
      1,
      {1, 1, 1});

  std::vector<HbLazyTensor> tensors = {
      GetHbLazyTensor(out1), GetHbLazyTensor(out2), GetHbLazyTensor(out3)};

  std::vector<ir::NodePtr> a{tensors[0].CurrentIrValue().mp_node};
  std::vector<int> indices1{0, 1, 2};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=[1, 1]]")
      ->check("prim::Constant[value=[0, 0]]")
      ->check("prim::Constant[value=[1, 1]]")
      ->check("prim::Constant[value=0]")
      ->check("prim::Constant[value=[0, 0]]")
      ->check("prim::Constant[value=1]")
      ->check("prim::Constant[value=[1, 1, 1]]")
      ->check_count("aten::convolution_backward_overrideable", 1)
      ->run(*hlexec->get_graph());
}

TEST_F(LazyConvKernelTest, ConvExecTest) {
  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({5, 4, 3, 3}, torch::dtype(torch::kFloat)); // kchw
  auto exp = torch::conv2d(in, wt, {}, 1, 0, 1, 1);

  auto h_in = in.to(torch::kHABANA);
  auto wt_hwck = wt.permute({2, 3, 1, 0}).contiguous();
  auto h_wt = wt_hwck.to(torch::kHABANA);

  torch::Tensor result = torch::conv2d(h_in, h_wt, {}, 1, 0, 1, 1);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}