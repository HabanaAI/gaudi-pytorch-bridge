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

class LazySoftmaxKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor input = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHABANA);
  int dim = 0;
  torch::Tensor hout = torch::log_softmax(hinput, dim);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto hout1 = hout.to(torch::kCPU);

  auto cout = torch::log_softmax(input, dim);

  EXPECT_EQ(allclose(hout1, cout), true);
  unsetenv("PT_HPU_LAZY_MODE");

}

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTestBackward) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor input = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor grad = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor output = torch::rand({64,10}, torch::requires_grad(false));

  torch::Tensor hinput = input.to(torch::kHABANA);
  torch::Tensor hgrad = grad.to(torch::kHABANA);
  torch::Tensor houtput = output.to(torch::kHABANA);

  int dim = 0;
  auto hout_backward = torch::_log_softmax_backward_data(hgrad, houtput, dim, hinput);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto hout2_back = hout_backward.to(torch::kCPU);

  auto cout_back = _log_softmax_backward_data(grad, output, dim, input);

  EXPECT_EQ(allclose(hout2_back, cout_back), true);
  unsetenv("PT_HPU_LAZY_MODE");

}