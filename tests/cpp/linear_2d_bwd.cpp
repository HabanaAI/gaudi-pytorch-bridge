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

auto linear_test = [](std::vector<int64_t> in_shape /* shape upto n-1*/,
                      bool bias_requied,
                      bool dynamic) {
  int out_features = 5;
  int in_features = 4;

  for (int i = 0; i <= 2 * dynamic; i++) {
    out_features += i;
    in_features += i;
    in_shape.push_back(in_features);

    auto in = torch::randn(in_shape, torch::requires_grad());
    auto hin = in.to(torch::kHPU);
    auto wt = torch::randn(
        {out_features, in_features}, torch::requires_grad()); // ckhw
    auto hwt = wt.to(torch::kHPU);
    auto bias = torch::randn({out_features}, torch::requires_grad());
    auto hbias = bias.to(torch::kHPU);

    auto exp = torch::linear(in, wt, bias);
    auto exp_hpu = habana_lazy::linear_non2d_hpu_lazy(hin, hwt, hbias);

    auto grad_out = torch::ones_like(exp.detach());
    auto hgrad_out = grad_out.detach().to(torch::kHPU);

    exp.backward(grad_out);

    auto grad_in = in.grad();
    auto grad_wt = wt.grad();
    auto grad_bias = bias.grad();

    at::Tensor hgrad_in, hgrad_wt, hgrad_bias;
    std::array<bool, 3> mask{1, 1, bias_requied};
    std::tie(hgrad_in, hgrad_wt, hgrad_bias) =
        habana_lazy::linear_bwd_hpu_lazy(hin, hgrad_out, hwt, mask);

    auto hgrad_wt_cpu = hgrad_wt.to(torch::kCPU);
    auto hgrad_in_cpu = hgrad_in.to(torch::kCPU);

    EXPECT_EQ(allclose(grad_wt, hgrad_wt_cpu, 0.01, 0.01), true);
    in_shape.pop_back();
  }
};

TEST_F(LazyLinearBwdTest, LinearBwdTest2D) {
  linear_test({3}, 0, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest2DBias) {
  linear_test({3}, 1, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest2DDynamic) {
  linear_test({3}, 0, 1);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest2DBiasDynamic) {
  linear_test({3}, 1, 1);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest1D) {
  linear_test({}, 0, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest1DBias) {
  linear_test({}, 1, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest1DDynamic) {
  linear_test({}, 0, 1);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest1DBiasDynamic) {
  linear_test({}, 1, 1);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest3D) {
  linear_test({2, 3}, 0, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest3DBias) {
  linear_test({2, 3}, 1, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest3DDynamic) {
  linear_test({2, 3}, 0, 1);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest3DBiasDynamic) {
  linear_test({2, 3}, 1, 1);
}

TEST_F(LazyLinearBwdTest, LinearBwdTest4D) {
  linear_test({2, 4, 3}, 0, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest4DBias) {
  linear_test({2, 4, 3}, 1, 0);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest4DDynamic) {
  linear_test({2, 4, 3}, 0, 1);
}
TEST_F(LazyLinearBwdTest, LinearBwdTest4DBiasDynamic) {
  linear_test({2, 4, 3}, 1, 1);
}
