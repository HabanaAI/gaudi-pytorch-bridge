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
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyPoolKernelTest, MaxPoolBWDTest) {
  auto input_tensor =
      torch::arange(20, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 4, 5}); // nchw
  auto cpu_pool = torch::max_pool2d(input_tensor, 3, 1);
  auto cpu_out = torch::relu(cpu_pool);

  // fwd propga
  torch::Tensor tHabanaX = input_tensor.to(torch::kHABANA);
  auto outHabana1 = torch::max_pool2d_with_indices(tHabanaX, {3, 3}, {1,1}, {0, 0}, {1, 1}, true);
  torch::Tensor outHabana = torch::relu(std::get<0>(outHabana1));

  // bwd propga with dummy grad tensor
  auto grad_tensor =
      torch::arange(6, torch::dtype(torch::kFloat).requires_grad(true))
          .reshape({1, 1, 2, 3});
  torch::Tensor tHabanaG = grad_tensor.to(torch::kHABANA);
  outHabana.backward({tHabanaG}, false, true);

  auto out_cpu_lazy = outHabana.to(torch::kCPU);
  ASSERT_TRUE(torch::allclose(out_cpu_lazy, cpu_out));
}

