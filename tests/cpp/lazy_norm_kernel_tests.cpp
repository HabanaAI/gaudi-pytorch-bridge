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

class LazyNormKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyNormKernelTest, LayerNormForward) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  at::Tensor weight =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHABANA);
  at::Tensor bias =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHABANA);
  auto results =
      torch::native_layer_norm(tHabanaX, tWeight, tBias, 10, 480, 0.01);

  auto hl_result =
      std::make_shared<HbLazyTensor>(GetHbLazyTensor(std::get<0>(results)));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  std::vector<HbLazyTensor> tensors = {*hl_result};
  std::vector<int> indices = {0};

  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("aten::native_layer_norm")
      ->run(*exec.get_graph());
  // Match lazy IR graph
  auto out_string = IrGraphDumpUtil::ToText(a);
  EXPECT_EQ(
      out_string.find(
          "IR {\n  %0 = hpu::input()\n  %1 = hpu::input()\n  %2 = hpu::input()\n  %3 = aten::native_layer_norm(%2, %1, %0), M=10, N=480, EPS=0.01, ROOT=0\n}\n"),
      0);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyNormKernelTest, LayerNormBackward) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_grad =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHABANA);
  auto input =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tHabanaIn = input.to(torch::kHABANA);
  auto mean = torch::arange(1, torch::dtype(torch::kFloat).requires_grad(false))
                  .reshape({1, 1});
  auto var = torch::arange(1, torch::dtype(torch::kFloat).requires_grad(false))
                 .reshape({1, 1});
  torch::Tensor tHabanaMean = mean.to(torch::kHABANA);
  torch::Tensor tHabanaVar = var.to(torch::kHABANA);
  at::Tensor weight;
  auto results = torch::native_layer_norm_backward(
      tHabanaGrad,
      tHabanaIn,
      tHabanaMean,
      tHabanaVar,
      weight,
      1,
      48,
      {true, true, true});
  // Match lazy IR graph
  auto hl_result =
      std::make_shared<HbLazyTensor>(GetHbLazyTensor(std::get<0>(results)));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};

  std::vector<HbLazyTensor> tensors = {*hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("aten::native_layer_norm_backward")
      ->run(*exec.get_graph());

  auto out_string = IrGraphDumpUtil::ToText(a);
  EXPECT_EQ(
      out_string.find(
          "IR {\n  %0 = hpu::input()\n  %1 = hpu::input()\n  %2 = hpu::input()\n  %3 = hpu::input()\n  %4 = aten::native_layer_norm_backward(%3, %2, %1, %0), M=1, N=48, ROOT=0\n}\n"),
      0);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyNormKernelTest, LayerNormForwardExecute) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  at::Tensor weight =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tWeight = weight.to(torch::kHABANA);
  at::Tensor bias =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw;
  torch::Tensor tBias = bias.to(torch::kHABANA);
  auto results =
      torch::native_layer_norm(tHabanaX, tWeight, tBias, 10, 48, 0.01);
  auto hl_result =
      std::make_shared<HbLazyTensor>(GetHbLazyTensor(std::get<0>(results)));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});
  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
  unsetenv("PT_HPU_LAZY_MODE");
  auto results_cpu =
      torch::native_layer_norm(input_tensor, weight, bias, 10, 48, 0.01);
  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyNormKernelTest, LayerNormBackwardExecute) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_grad =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaGrad = input_grad.to(torch::kHABANA);
  auto input =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanaIn = input.to(torch::kHABANA);
  auto mean =
      torch::arange(10, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 1});
  auto var = torch::arange(10, torch::dtype(torch::kFloat).requires_grad(false))
                 .reshape({10, 1});
  torch::Tensor tHabanaMean = mean.to(torch::kHABANA);
  torch::Tensor tHabanaVar = var.to(torch::kHABANA);
  auto gamma =
      torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({1, 3, 4, 4}); // nchw
  torch::Tensor tGamma = gamma.to(torch::kHABANA);
  auto results = torch::native_layer_norm_backward(
      tHabanaGrad,
      tHabanaIn,
      tHabanaMean,
      tHabanaVar,
      tGamma,
      10,
      48,
      {true, true, true});
  // Match lazy IR graph
  auto hl_result =
      std::make_shared<HbLazyTensor>(GetHbLazyTensor(std::get<0>(results)));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<ir::NodePtr> a{ir_value.mp_node};
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});
  at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
  unsetenv("PT_HPU_LAZY_MODE");
  auto results_cpu = torch::native_layer_norm_backward(
      input_grad, input, mean, var, gamma, 10, 48, {true, true, true});
  at::Tensor result_cpu = std::get<0>(results_cpu);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}