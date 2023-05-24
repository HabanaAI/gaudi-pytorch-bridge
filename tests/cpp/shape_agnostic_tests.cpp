#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/jit_graph_cache.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;
using namespace at;

class ShapeAgnosticTest : public habana_lazy_test::LazyTest {
 protected:
  void SetUp() override {
    SetEagerMode();
    DisableRecipeCache();
    EnableShapeAgnostic();
    SetSeed();
  }

  void TearDown() override {
    RestoreRecipeCache();
    RestoreShapeAgnostic();
    RestoreMode();
  }
};

TEST_F(ShapeAgnosticTest, PermuteAdd) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3, 3});
    auto B = A.permute({0, 1, 2}).contiguous();
    auto C = B.add(1.0);
    auto hA = A.to(torch::kHPU);
    auto hB = hA.permute({0, 1, 2}).contiguous();
    auto hC = hB.add(1.0);

    EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);

    torch::Tensor D = torch::randn({6, 6, 6});
    auto E = D.permute({0, 1, 2}).contiguous();
    auto F = E.add(1.0);
    auto hD = D.to(torch::kHPU);
    auto hE = hD.permute({0, 1, 2}).contiguous();
    auto hF = hE.add(1.0);

    EXPECT_EQ(allclose(F, hF.cpu(), 0.001, 0.001), true);
  }
}

TEST_F(ShapeAgnosticTest, ConvRelu) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto input_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 3, 3}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto input_tensor_2 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 4, 4, 4}); // nchw
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);

    auto weight_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({3, 3, 3, 1}); // hwck
    auto weight_tensor_2 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({4, 4, 4, 1}); // hwck

    torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
    torch::Tensor tHabanaW_2 = weight_tensor_2.to(torch::kHPU);

    torch::Tensor outConv =
        torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outConv_2 = torch::conv2d(
        tHabanaX_2, tHabanaW_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor outhpu = torch::relu(outConv);
    torch::Tensor outhpu_2 = torch::relu(outConv_2);

    torch::Tensor out = outhpu.to(torch::kCPU);
    torch::Tensor out_2 = outhpu_2.to(torch::kCPU);
    torch::Tensor out_conv = outConv.to(torch::kCPU);
    torch::Tensor out_conv_2 = outConv_2.to(torch::kCPU);

    torch::Tensor outConv1 = torch::conv2d(
        input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu = torch::relu(outConv1);

    torch::Tensor outConv2 = torch::conv2d(
        input_tensor_2, weight_tensor_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu_2 = torch::relu(outConv2);

    EXPECT_EQ(allclose(out_conv, outConv1, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_conv_2, outConv2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_2, outcpu_2, 0.01, 0.01), true);
  }
}

// To validate permute information as part of JIT/SAG key calculation
// 1st and 2nd relu has input with real permute while 3rd relu does not
// have any permute on the input so 3rd relu should cause a JIT/SAG cache
// miss.
TEST_F(ShapeAgnosticTest, ConvReluRelu) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    // Disabling the number of cache entries check for now as the same
    // test is being also called for the lazy frontend as well and there
    // the cache class instance is different than PT2.0 eager.
#if 0
    habana::OptimizedJitGraphCache::GetOptimizedJitCache().BackupCache();
    habana::OptimizedJitGraphCache::GetOptimizedJitCache().Clear();
    size_t num_cache_entries_start =
        habana::OptimizedJitGraphCache::GetOptimizedJitCache().CacheSize();
#endif

    auto input_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 3, 3}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    auto input_tensor_2 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 4, 4, 4}); // nchw
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);
    auto input_tensor_3 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 4, 4, 4}); // nchw
    torch::Tensor tHabanaX_3 = input_tensor_3.to(torch::kHPU);

    auto weight_tensor =
        torch::arange(27, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({3, 3, 3, 1}); // hwck
    auto weight_tensor_2 =
        torch::arange(64, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({4, 4, 4, 1}); // hwck

    torch::Tensor tHabanaW = weight_tensor.to(torch::kHPU);
    torch::Tensor tHabanaW_2 = weight_tensor_2.to(torch::kHPU);

    torch::Tensor outConv =
        torch::conv2d(tHabanaX, tHabanaW, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outConv_2 = torch::conv2d(
        tHabanaX_2, tHabanaW_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);

    torch::Tensor outhpu = torch::relu(outConv);
    torch::Tensor outhpu_2 = torch::relu(outConv_2);
    torch::Tensor outhpu_3 = torch::relu(tHabanaX_3);

    torch::Tensor out = outhpu.to(torch::kCPU);
    torch::Tensor out_2 = outhpu_2.to(torch::kCPU);
    torch::Tensor out_3 = outhpu_3.to(torch::kCPU);
    torch::Tensor out_conv = outConv.to(torch::kCPU);
    torch::Tensor out_conv_2 = outConv_2.to(torch::kCPU);

    torch::Tensor outConv1 = torch::conv2d(
        input_tensor, weight_tensor, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu = torch::relu(outConv1);

    torch::Tensor outConv2 = torch::conv2d(
        input_tensor_2, weight_tensor_2, {}, {1}, at::IntArrayRef{0}, {1}, 1);
    torch::Tensor outcpu_2 = torch::relu(outConv2);

    torch::Tensor outcpu_3 = torch::relu(input_tensor_3);

#if 0
    size_t num_cache_entries_end =
        habana::OptimizedJitGraphCache::GetOptimizedJitCache().CacheSize();
    size_t num_cache_entries = num_cache_entries_end - num_cache_entries_start;
    habana::OptimizedJitGraphCache::GetOptimizedJitCache().RestoreCache();
    habana::OptimizedJitGraphCache::GetOptimizedJitCache().ClearBackupCache();
#endif

    EXPECT_EQ(allclose(out_conv, outConv1, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_conv_2, outConv2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_2, outcpu_2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_3, outcpu_3, 0.01, 0.01), true);
#if 0
    EXPECT_EQ(num_cache_entries, 3);
#endif
  }
}
