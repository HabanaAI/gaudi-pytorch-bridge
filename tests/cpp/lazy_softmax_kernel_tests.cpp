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
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class LazySoftmaxKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTest) {
  torch::Tensor input = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHABANA);
  int dim = 0;
  torch::Tensor hout = torch::log_softmax(hinput, dim);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout1 = hout.to(torch::kCPU);

  auto cout = torch::log_softmax(input, dim);

  EXPECT_EQ(allclose(hout1, cout), true);
}

TEST_F(LazySoftmaxKernelTest, LogSoftMaxTestBackward) {
  torch::Tensor input = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor grad = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor output = torch::rand({64,10}, torch::requires_grad(false));

  torch::Tensor hinput = input.to(torch::kHABANA);
  torch::Tensor hgrad = grad.to(torch::kHABANA);
  torch::Tensor houtput = output.to(torch::kHABANA);

  int dim = 0;
  auto hout_backward = torch::_log_softmax_backward_data(hgrad, houtput, dim, hinput);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout2_back = hout_backward.to(torch::kCPU);

  auto cout_back = _log_softmax_backward_data(grad, output, dim, input);

  EXPECT_EQ(allclose(hout2_back, cout_back), true);

}

TEST_F(LazySoftmaxKernelTest, SoftMaxTest) {
  torch::Tensor input = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor hinput = input.to(torch::kHABANA);
  int dim = 0;
  torch::Tensor hout = torch::_softmax(hinput, dim, false);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout1 = hout.to(torch::kCPU);

  auto cout = torch::_softmax(input, dim, false);

  EXPECT_EQ(allclose(hout1, cout), true);
}

TEST_F(LazySoftmaxKernelTest, SoftMaxTestBackward) {
  torch::Tensor input = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor grad = torch::rand({64,10}, torch::requires_grad(false));
  torch::Tensor output = torch::rand({64,10}, torch::requires_grad(false));

  torch::Tensor hinput = input.to(torch::kHABANA);
  torch::Tensor hgrad = grad.to(torch::kHABANA);
  torch::Tensor houtput = output.to(torch::kHABANA);

  int dim = 0;
  auto hout_backward = torch::_softmax_backward_data(hgrad, houtput, dim, hinput);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hout_backward)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hout2_back = hout_backward.to(torch::kCPU);

  auto cout_back = _softmax_backward_data(grad, output, dim, input);

  EXPECT_EQ(allclose(hout2_back, cout_back), true);
}
