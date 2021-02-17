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

  std::vector<at::Tensor> input_list{h_grad_output, hinput, hweight};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=[1, 1]]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=[0, 0]]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=0]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=1]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=[True, True, True]]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check_count("aten::convolution_backward_overrideable", 1)
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

TEST_F(LazyConvKernelTest, ConvTranspose2dTest) {
  auto in = torch::randn({64, 4, 28, 28}, torch::dtype(torch::kFloat)); // nchw
  auto wt = torch::randn({4, 5, 3, 3}, torch::dtype(torch::kFloat)); // ckhw
  auto bias = torch::randn({5}, torch::dtype(torch::kFloat)); // k
  auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

  auto h_in = in.to(torch::kHABANA);
  auto wt_hwck = wt.permute({2, 3, 1, 0}).contiguous();
  auto h_wt = wt_hwck.to(torch::kHABANA);

  torch::Tensor result = torch::conv_transpose2d(h_in, h_wt, {}, 1, 0, 0, 1, 1);
  Tensor out = result.to(kCPU);
  EXPECT_EQ(allclose(out, exp, 0.01, 0.01), true);
}

TEST_F(LazyConvKernelTest, ConvTranspose2dBwdTest) {
  auto in = torch::randn({64, 4, 28, 28}, torch::requires_grad()); // nchw
  auto hin = in.to(torch::kHABANA);
  auto wt = torch::randn({4, 5, 3, 3}, torch::requires_grad()); // ckhw
  auto wt_hwck = wt.detach().permute({2, 3, 1, 0}).contiguous();
  auto hwt = wt_hwck.to(torch::kHABANA);
  auto bias = torch::randn({5}, torch::requires_grad()); // k
  auto exp = torch::conv_transpose2d(in, wt, {}, 1, 0, 0, 1, 1);

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHABANA);
  exp.backward(grad_out);
  auto grad_in = in.grad();
  auto grad_wt = wt.grad();

  Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 0};
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) = convolution_backward_hpu_lazy(
      hgrad_out, hin, hwt, {1, 1}, {0, 0}, {1, 1}, true, {0, 0}, 1, mask);

  // TBD: aten::backward is not handled by lazy mode, therefore this is
  // not working. This code can be restored when that is fixed.
  /*auto result = torch::conv_transpose2d(hin, hwt, {}, 1, 0, 0, 1, 1);
  result.backward(hgrad_out);
  auto hgrad_in = hin.grad();
  auto hgrad_wt = hwt.grad();*/

  // without explicit stepmarker here. DMA for hgrad_in tensor gets messed up
  // most likely due to 2 outputs from backward op. TBD: remove this once issue
  // is debugged and fixed.
  HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  EXPECT_EQ(allclose(grad_in, hgrad_in_cpu, 0.01, 0.01), true);
  EXPECT_EQ(
      allclose(grad_wt, hgrad_wt_cpu.permute({3, 2, 0, 1}), 0.01, 0.01), true);
}