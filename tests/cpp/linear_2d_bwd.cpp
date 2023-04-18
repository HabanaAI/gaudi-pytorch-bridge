#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
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
class LazyLinearBwdTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyLinearBwdTest, LinearBwdTest) {
  int out_features = 5;
  int in_features = 4;
  int m = 2;
  int n = 2;

  auto in = torch::randn({3, in_features}, torch::requires_grad());
  auto hin = in.to(torch::kHPU);
  auto wt =
      torch::randn({out_features, in_features}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features}, torch::requires_grad());
  auto hbias = bias.to(torch::kHPU);

  auto exp = torch::linear(in, wt, bias);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << "exp_cpu" << exp.cpu() << std::endl;
  auto exp_hpu = habana_lazy::linear_non2d_hpu_lazy(hin, hwt, hbias);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << "exp_hpu" << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in = in.grad();
  auto grad_wt = wt.grad();
  auto grad_bias = bias.grad();

  at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 1};
  std::cout << grad_out << std::endl;
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
      habana_lazy::linear_bwd_hpu_lazy(hin, hgrad_out, hwt, mask);

  // HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  std::cout << "grad_wt" << grad_wt << std::endl;
  std::cout << "hgrad_wt_cpu" << hgrad_wt_cpu << std::endl;
  std::cout << "grad_in" << grad_in << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in << std::endl;
  std::cout << "bias_in_cpu" << grad_bias << std::endl;
  std::cout << "hbias_in_cpu" << hgrad_bias << std::endl;

  EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest3d) {
  int out_features = 5;
  int in_features = 4;
  int m = 2;
  int n = 2;

  auto in = torch::randn({2, 3, in_features}, torch::requires_grad());
  auto hin = in.to(torch::kHPU);
  auto wt =
      torch::randn({out_features, in_features}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features}, torch::requires_grad());
  auto hbias = bias.to(torch::kHPU);

  auto exp = torch::linear(in, wt, bias);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << "exp_cpu" << exp.cpu() << std::endl;
  auto exp_hpu = habana_lazy::linear_non2d_hpu_lazy(hin, hwt, hbias);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << "exp_hpu" << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in = in.grad();
  auto grad_wt = wt.grad();
  auto grad_bias = bias.grad();

  at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 1};
  std::cout << grad_out << std::endl;
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
      habana_lazy::linear_bwd_hpu_lazy(hin, hgrad_out, hwt, mask);

  // hgrad_bias = hgrad_out.sum_to_size(hwt.sizes().vec()[0]);
  // hgrad_in = torch::matmul(grad_out, hwt);
  // auto reshaped_in = torch::reshape(hin, {6, in_features});
  // auto reshaped_out = torch::reshape(hgrad_out, {6, out_features});
  // hgrad_wt = torch::matmul(torch::transpose(reshaped_out, 0, 1),
  // reshaped_in); HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  std::cout << "grad_wt" << grad_wt << std::endl;
  std::cout << "hgrad_wt_cpu" << hgrad_wt_cpu << std::endl;
  std::cout << "grad_in" << grad_in << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in << std::endl;
  std::cout << "bias_in_cpu" << grad_bias << std::endl;
  std::cout << "hbias_in_cpu" << hgrad_bias << std::endl;

  EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest4d) {
  int out_features = 5;
  int in_features = 4;
  int m = 2;
  int n = 2;

  auto in = torch::randn({2, 4, 3, in_features}, torch::requires_grad());
  auto hin = in.to(torch::kHPU);
  auto wt =
      torch::randn({out_features, in_features}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features}, torch::requires_grad());
  auto hbias = bias.to(torch::kHPU);

  auto exp = torch::linear(in, wt, bias);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << "exp_cpu" << exp.cpu() << std::endl;
  auto exp_hpu = habana_lazy::linear_non2d_hpu_lazy(hin, hwt, hbias);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << "exp_hpu" << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in = in.grad();
  auto grad_wt = wt.grad();
  auto grad_bias = bias.grad();

  at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 1};
  std::cout << grad_out << std::endl;
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
      habana_lazy::linear_bwd_hpu_lazy(hin, hgrad_out, hwt, mask);

  // hgrad_bias = hgrad_out.sum_to_size(hwt.sizes().vec()[0]);
  // hgrad_in = torch::matmul(grad_out, hwt);
  // auto reshaped_in = torch::reshape(hin, {6, in_features});
  // auto reshaped_out = torch::reshape(hgrad_out, {6, out_features});
  // hgrad_wt = torch::matmul(torch::transpose(reshaped_out, 0, 1),
  // reshaped_in); HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  std::cout << "grad_wt" << grad_wt << std::endl;
  std::cout << "hgrad_wt_cpu" << hgrad_wt_cpu << std::endl;
  std::cout << "grad_in" << grad_in << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in << std::endl;
  std::cout << "bias_in_cpu" << grad_bias << std::endl;
  std::cout << "hbias_in_cpu" << hgrad_bias << std::endl;

  EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest5d) {
  int out_features = 5;
  int in_features = 4;
  int m = 2;
  int n = 2;

  auto in = torch::randn({2, 4, 3, 3, in_features}, torch::requires_grad());
  auto hin = in.to(torch::kHPU);
  auto wt =
      torch::randn({out_features, in_features}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features}, torch::requires_grad());
  auto hbias = bias.to(torch::kHPU);

  auto exp = torch::linear(in, wt, bias);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << "exp_cpu" << exp.cpu() << std::endl;
  auto exp_hpu = habana_lazy::linear_non2d_hpu_lazy(hin, hwt, hbias);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << "exp_hpu" << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in = in.grad();
  auto grad_wt = wt.grad();
  auto grad_bias = bias.grad();

  at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 1};
  std::cout << grad_out << std::endl;
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
      habana_lazy::linear_bwd_hpu_lazy(hin, hgrad_out, hwt, mask);

  // hgrad_bias = hgrad_out.sum_to_size(hwt.sizes().vec()[0]);
  // hgrad_in = torch::matmul(grad_out, hwt);
  // auto reshaped_in = torch::reshape(hin, {6, in_features});
  // auto reshaped_out = torch::reshape(hgrad_out, {6, out_features});
  // hgrad_wt = torch::matmul(torch::transpose(reshaped_out, 0, 1),
  // reshaped_in); HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  std::cout << "grad_wt" << grad_wt << std::endl;
  std::cout << "hgrad_wt_cpu" << hgrad_wt_cpu << std::endl;
  std::cout << "grad_in" << grad_in << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in << std::endl;
  std::cout << "bias_in_cpu" << grad_bias << std::endl;
  std::cout << "hbias_in_cpu" << hgrad_bias << std::endl;

  EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest1d) {
  int out_features = 5;
  int in_features = 4;
  int m = 2;
  int n = 2;

  auto in = torch::randn({in_features}, torch::requires_grad());
  auto hin = in.to(torch::kHPU);
  auto wt =
      torch::randn({out_features, in_features}, torch::requires_grad()); // ckhw
  auto hwt = wt.to(torch::kHPU);
  auto bias = torch::randn({out_features}, torch::requires_grad());

  auto exp = torch::linear(in, wt, bias);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << "exp_cpu" << exp.cpu() << std::endl;
  auto exp_hpu = torch::linear(hin, hwt);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << "exp_hpu" << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in = in.grad();
  auto grad_wt = wt.grad();
  auto grad_bias = bias.grad();

  at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
  std::array<bool, 3> mask{1, 1, 1};
  std::cout << grad_out << std::endl;
  std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
      habana_lazy::linear_bwd_hpu_lazy(hin, hgrad_out, hwt, mask);

  // hgrad_bias = hgrad_out.sum_to_size(hwt.sizes().vec()[0]);
  // hgrad_in = torch::matmul(grad_out, hwt);
  // auto reshaped_in = torch::reshape(hin, {6, in_features});
  // auto reshaped_out = torch::reshape(hgrad_out, {6, out_features});
  // hgrad_wt = torch::matmul(torch::transpose(reshaped_out, 0, 1),
  // reshaped_in); HbLazyTensor::StepMarker({});

  auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
  auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);
  std::cout << "grad_wt" << grad_wt << std::endl;
  std::cout << "hgrad_wt_cpu" << hgrad_wt_cpu << std::endl;
  std::cout << "grad_in" << grad_in << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in << std::endl;
  std::cout << "bias_in_cpu" << grad_bias << std::endl;
  std::cout << "hbias_in_cpu" << hgrad_bias << std::endl;

  EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
}

TEST_F(LazyLinearBwdTest, MatmulTest) {
  auto in_a = torch::randn({3, 2, 3, 4}, torch::requires_grad());
  auto hin_a = in_a.to(torch::kHPU);
  auto in_b = torch::randn({3, 2, 4, 5}, torch::requires_grad());
  auto hin_b = in_b.to(torch::kHPU);

  auto exp = torch::matmul(in_a, in_b);
  std::cout << exp.sizes().vec() << std::endl;
  std::cout << "exp_cpu" << exp.cpu() << std::endl;
  auto exp_hpu = habana_lazy::matmul_hpu_lazy(hin_a, hin_b);
  std::cout << exp_hpu.sizes().vec() << std::endl;
  std::cout << "exp_hpu" << exp_hpu.cpu() << std::endl;

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in_a = in_a.grad();
  auto grad_in_b = in_b.grad();

  at::Tensor hgrad_in_a, hgrad_in_b;
  std::cout << grad_out << std::endl;
  std::tie(hgrad_in_a, hgrad_in_b) =
      habana_lazy::matmul_backward_hpu_lazy(hgrad_out, hin_a, hin_b);

  // HbLazyTensor::StepMarker({});

  auto hgrad_in_a_cpu = hgrad_in_a.to(torch::kCPU);
  auto hgrad_in_b_cpu = hgrad_in_b.to(torch::kCPU);
  std::cout << "grad_in_a" << grad_in_a << std::endl;
  std::cout << "hgrad_in_a" << hgrad_in_a << std::endl;
  std::cout << "grad_in_b" << grad_in_b << std::endl;
  std::cout << "hgrad_in_cpu" << hgrad_in_b << std::endl;

  EXPECT_EQ(allclose(grad_in_a, hgrad_in_a, 0.01, 0.01), true);
}

TEST_F(LazyLinearBwdTest, MatmulTest2d) {
  auto in_a = torch::randn({3, 4}, torch::requires_grad());
  auto hin_a = in_a.to(torch::kHPU);
  auto in_b = torch::randn({4, 5}, torch::requires_grad());
  auto hin_b = in_b.to(torch::kHPU);

  auto exp = torch::matmul(in_a, in_b);
  auto exp_hpu = habana_lazy::matmul_hpu_lazy(hin_a, hin_b);

  auto grad_out = torch::ones_like(exp.detach());
  auto hgrad_out = grad_out.detach().to(torch::kHPU);

  exp.backward(grad_out);

  auto grad_in_a = in_a.grad();
  auto grad_in_b = in_b.grad();

  at::Tensor hgrad_in_a, hgrad_in_b;
  std::tie(hgrad_in_a, hgrad_in_b) =
      habana_lazy::matmul_backward_hpu_lazy(hgrad_out, hin_a, hin_b);

  // HbLazyTensor::StepMarker({});

  auto hgrad_in_a_cpu = hgrad_in_a.to(torch::kCPU);
  auto hgrad_in_b_cpu = hgrad_in_b.to(torch::kCPU);

  EXPECT_EQ(allclose(grad_in_a, hgrad_in_a, 0.01, 0.01), true);
}
