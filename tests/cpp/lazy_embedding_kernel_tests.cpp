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

void embedding_bwd_test(
    int num_emb = 10,
    int dimension = 3,
    int padding_idx = -1,
    bool is_dynamic = false) {
  for (int i = 0; i <= is_dynamic * 1; i++) {
    int num_embeddings = num_emb;
    int dim = dimension;
    auto tindices =
        torch::randint(0, 9, at::IntArrayRef({3 + i, 4 + i, 5}), torch::kInt64);
    auto htindices = tindices.to(torch::kHPU);

    auto tweights = torch::randn({num_embeddings, dim}, torch::requires_grad());
    auto htweights = tweights.to(torch::kHPU);
    auto c_out =
        torch::embedding(tweights, tindices, padding_idx, false, false);
    auto h_out = c_out.to(torch::kCPU);

    auto grad_out = torch::ones_like(c_out);
    auto hgrad_out = grad_out.to(torch::kHPU);
    c_out.backward(grad_out);

    auto weight_grad = tweights.grad();
    auto hweight_grad = torch::embedding_dense_backward(
        hgrad_out, htindices, num_emb, padding_idx, false);
    EXPECT_EQ(
        allclose(weight_grad, hweight_grad.to(torch::kCPU), 0.01, 0.01), true);
  }
}

TEST_F(LazyEmbeddingKernelTest, EmbeddingBwdTest) {
  embedding_bwd_test(10, 3, -1);
}
TEST_F(LazyEmbeddingKernelTest, EmbeddingBwdDynamicTest) {
  embedding_bwd_test(10, 3, -1, true);
}
TEST_F(LazyEmbeddingKernelTest, EmbeddingBwddPaddingIdxTest) {
  embedding_bwd_test(10, 3, 2);
}
TEST_F(LazyEmbeddingKernelTest, EmbeddingBwdPaddingIdxDynamicTest) {
  embedding_bwd_test(10, 3, 2, true);
}
