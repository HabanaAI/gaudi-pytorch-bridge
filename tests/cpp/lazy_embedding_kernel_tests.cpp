#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;
using namespace at;

class LazyEmbeddingKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyEmbeddingKernelTest, EmbeddingTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  auto tindices = torch::randint(9, 10, at::IntArrayRef({10}), torch::kInt64);
  torch::Tensor htindices = tindices.to(torch::kHPU);

  Tensor tweights = torch::randn({10, 2});
  torch::Tensor htweights = tweights.to(torch::kHPU);
  auto hembed = torch::embedding(htweights, htindices, -1, false, false);
  auto hout = hembed.to(torch::kCPU);
  auto cout = torch::embedding(tweights, tindices, -1, false, false);
  EXPECT_EQ(allclose(hout, cout), true);

  // [ToDo] Backward not ready yet
  // Tensor tgrad = torch::randn({10, 2});
  // torch::Tensor htgrad = tgrad.to(torch::kHPU);
  // auto hembed_bwd = torch::embedding_dense_backward(htgrad, htindices, 10,
  // -1, false); auto hout_bwd = hembed_bwd.to(torch::kCPU); auto cout_bwd =
  // torch::embedding_dense_backward(tgrad, tindices, 10, -1, false);
  // EXPECT_EQ(allclose(hout_bwd, cout_bwd), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}
