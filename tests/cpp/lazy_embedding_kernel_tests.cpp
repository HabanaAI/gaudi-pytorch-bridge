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

class LazyEmbeddingKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyEmbeddingKernelTest, EmbeddingTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 0);

  auto tindices = torch::randint(9, 10, at::IntArrayRef({10}), torch::kInt64);
  torch::Tensor htindices = tindices.to(torch::kHABANA);

  Tensor tweights = torch::randn({10, 2});
  torch::Tensor htweights = tweights.to(torch::kHABANA);
  auto hembed = torch::embedding(htweights, htindices, -1, false, false);
  auto hout = hembed.to(torch::kCPU);
  auto cout = torch::embedding(tweights, tindices, -1, false, false);
  EXPECT_EQ(allclose(hout, cout), true);

  // [ToDo] Backward not ready yet
  // Tensor tgrad = torch::randn({10, 2});
  // torch::Tensor htgrad = tgrad.to(torch::kHABANA);
  // auto hembed_bwd = torch::embedding_dense_backward(htgrad, htindices, 10,
  // -1, false); auto hout_bwd = hembed_bwd.to(torch::kCPU); auto cout_bwd =
  // torch::embedding_dense_backward(tgrad, tindices, 10, -1, false);
  // EXPECT_EQ(allclose(hout_bwd, cout_bwd), true);

  unsetenv("PT_HPU_LAZY_MODE");
}