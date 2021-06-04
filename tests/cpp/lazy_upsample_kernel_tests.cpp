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

using namespace habana_lazy;
using namespace at;

class LazyUpsampleKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyUpsampleKernelTest, DISABLED_UpsampleNearestTest) {
  torch::Tensor tensor = torch::randn({3, 1, 5, 5});
  torch::Tensor tHabana = tensor.to(torch::kHABANA);
  c10::ArrayRef<double> scale_factors = {2.0, 2.0};
  auto outHabana = torch::upsample_nearest2d(tHabana, {}, scale_factors);
  auto out = torch::upsample_nearest2d(tensor, {}, scale_factors);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUpsampleKernelTest, UpsampleNearest2Test) {
  torch::Tensor tensor = torch::randn({3, 1, 5, 5});
  torch::Tensor tHabana = tensor.to(torch::kHABANA);
  c10::ArrayRef<int64_t> sizes = {10, 15};
  auto outHabana = torch::upsample_nearest2d(tHabana, sizes, {});
  auto out = torch::upsample_nearest2d(tensor, sizes, {});
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyUpsampleKernelTest, UpsampleBackwardTest) {
  torch::manual_seed(0);
  auto upsample_test = [](c10::IntArrayRef size1) {
    auto mat1 = torch::randn(size1);
    auto mat1_h = mat1.to(torch::kHABANA);
    mat1.set_requires_grad(true);

    auto out = torch::upsample_nearest2d(mat1, {8, 21});
    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHABANA);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();

    torch::Tensor grad_mat1_h;
    c10::ArrayRef<double> scale_factors = {2.0, 2.0};
    c10::IntArrayRef out_size = {8, 21};
    grad_mat1_h = upsample_nearest2d_backward_hpu_lazy(
        grad_out_h, out_size, size1, scale_factors);
    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
  };
  upsample_test({1, 1, 4, 7});
}
