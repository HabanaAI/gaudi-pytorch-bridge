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

class BaseSettings : public habana_lazy_test::EnvHelper {
 public:
  void Create(bool enable_shape_agnostic = true) {
    habana::HABANAGuardImpl device_guard;
    device_guard.getDevice();
    SetEagerMode();
    DisableRecipeCache();
    EnableShapeAgnostic(enable_shape_agnostic);
    SetSeed();
  }
  void Finish() {
    RestoreRecipeCache();
    RestoreShapeAgnostic();
    RestoreMode();
  }
};

class ShapeAgnosticTest : public BaseSettings, public ::testing::Test {
 protected:
  void SetUp() override {
    Create();
  }

  void TearDown() override {
    Finish();
  }
};

class ShapeAgnosticOrNormalFlowTest : public BaseSettings,
                                      public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    Create(GetParam());
  }

  void TearDown() override {
    Finish();
  }
};

INSTANTIATE_TEST_SUITE_P(
    ConvPermutation,
    ShapeAgnosticOrNormalFlowTest,
    ::testing::Values(true, false));

TEST_F(ShapeAgnosticTest, PermuteAdd) {
  auto& device = habana::HPURegistrar::get_device();
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

TEST_P(ShapeAgnosticOrNormalFlowTest, ConvRelu) {
  habana::OptimizedJitGraphCache::GetOptimizedJitCache().Clear();
  auto& device = habana::HPURegistrar::get_device();
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
TEST_P(ShapeAgnosticOrNormalFlowTest, ConvReluRelu) {
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    // Disabling the number of cache entries check for now as the same
    // test is being also called for the lazy frontend as well and there
    // the cache class instance is different than PT2.0 eager.
    habana::OptimizedJitGraphCacheBackup make_cache_backup;

    size_t num_cache_entries_start =
        habana::OptimizedJitGraphCache::GetOptimizedJitCache().CacheSize();

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

    size_t num_cache_entries_end =
        habana::OptimizedJitGraphCache::GetOptimizedJitCache().CacheSize();
    size_t num_cache_entries = num_cache_entries_end - num_cache_entries_start;

    EXPECT_EQ(allclose(out_conv, outConv1, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_conv_2, outConv2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out, outcpu, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_2, outcpu_2, 0.01, 0.01), true);
    EXPECT_EQ(allclose(out_3, outcpu_3, 0.01, 0.01), true);
    EXPECT_EQ(num_cache_entries, 4);
  }
}

TEST_F(ShapeAgnosticTest, ScalarAdd) {
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    torch::Tensor A = torch::randn({3, 3, 3}, torch::dtype(torch::kFloat))
                          .to(torch::dtype(torch::kLong));
    auto hA = A.to(torch::kHPU);
    A = A.add_(1);
    hA = hA.add_(1);

    EXPECT_EQ(allclose(A, hA.cpu(), 0, 0), true);

    torch::Tensor B = torch::randn({6, 6, 6}, torch::dtype(torch::kFloat))
                          .to(torch::dtype(torch::kLong));
    auto hB = B.to(torch::kHPU);
    B = B.add_(2);
    hB = hB.add_(2);

    EXPECT_EQ(allclose(B, hB.cpu(), 0, 0), true);
  }
}

TEST_F(ShapeAgnosticTest, ResizeZST) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor input = torch::randn(
          {in_shapes[i], in_shapes[i]}, torch::dtype(torch::kFloat));
      // empty zst tensors
      torch::Tensor values = torch::empty(0, torch::dtype(torch::kFloat));
      torch::Tensor indices = torch::empty(0, torch::dtype(torch::kLong));

      auto input_hpu = input.to(torch::kHPU);
      auto indices_hpu = indices.to(torch::kHPU);
      auto values_hpu = values.to(torch::kHPU);

      torch::zero_(values);
      torch::zero_(indices);
      torch::median_outf(input, 0, false, values, indices);

      // InplaceOut -> torch.median(input, 0, out=(values, indices))
      // resize is called on zst tensors
      torch::zero_(values_hpu);
      torch::zero_(indices_hpu);
      torch::median_outf(input_hpu, 0, false, values_hpu, indices_hpu);

      EXPECT_EQ(allclose(indices, indices_hpu.cpu(), 0.01, 0.01), true);
      EXPECT_EQ(allclose(values, values_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, CopyD2HView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor in0 =
          torch::arange(in_shapes[i], torch::dtype(torch::kInt32));
      // non contiguous view
      auto in1 = in0.as_strided({4, 4}, {1, 4}, 0);

      // .cpu() -> calls strided_view + copy D2H
      auto in0_hpu = in0.to(torch::kHPU);
      auto in1_cpu = in0_hpu.as_strided({4, 4}, {1, 4}, 0).cpu();

      EXPECT_EQ(allclose(in1, in1_cpu, 0, 0), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, CopyH2DView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      // input is non contiguous view
      torch::Tensor in0 =
          torch::arange(in_shapes[i], torch::dtype(torch::kInt32))
              .as_strided({4, 4}, {1, 4}, 0);

      // .hpu() -> calls copy H2D + strided_insert
      auto in0_hpu = in0.to(torch::kHPU);

      // .cpu() -> calls strided_view + copy D2H
      auto in0_cpu = in0_hpu.to(torch::kCPU);

      EXPECT_EQ(allclose(in0, in0_cpu, 0, 0), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, CopyD2DView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      // input is non contiguous view
      torch::Tensor in0 =
          torch::arange(in_shapes[i], torch::dtype(torch::kInt32))
              .as_strided({4, 4}, {1, 4}, 0);

      // .hpu() -> calls copy H2D + strided_insert
      auto in0_hpu = in0.to(torch::kHPU);

      // clone() -> calls D2D copy with strided_insert
      auto in1_hpu = in0_hpu.clone();

      EXPECT_EQ(allclose(in0, in1_hpu.cpu(), 0, 0), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, CopyD2DOffsetView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    std::vector<int> in_offsets = {4, 8};

    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor in0 =
          torch::arange(in_shapes[i], torch::dtype(torch::kInt32));
      auto in0_hpu = in0.to(torch::kHPU);

      // contiguous view with offset
      auto in1 = in0.as_strided({in_offsets[i]}, {1}, in_offsets[i]);
      auto in1_hpu = in0_hpu.as_strided({in_offsets[i]}, {1}, in_offsets[i]);

      // clone copies to new tensor w.r.t offset
      // shape agnostic cache hit for copy takes care of new offset
      auto in2_hpu = in1_hpu.clone();

      EXPECT_EQ(allclose(in1, in2_hpu.cpu(), 0, 0), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, AddView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor in0 = torch::randn({in_shapes[i]});
      auto in0_hpu = in0.to(torch::kHPU);

      auto in1 = in0.as_strided({4, 4}, {1, 4}, 0);
      in1.add_(1.0);

      // graph -> strided_view + add + strided_insert
      auto in1_hpu = in0_hpu.as_strided({4, 4}, {1, 4}, 0);
      in1_hpu.add_(1.0);

      EXPECT_EQ(allclose(in1, in1_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, Gelu) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      auto input = torch::randn({in_shapes[i]}, torch::dtype(torch::kFloat));
      auto input_hpu = input.to(torch::kHPU);

      auto output = torch::empty(0, torch::dtype(torch::kFloat));
      auto output_hpu = output.to(torch::kHPU);

      torch::gelu_outf(input, "tanh", output);
      torch::gelu_outf(input_hpu, "tanh", output_hpu);

      EXPECT_EQ(allclose(output, output_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, GeluView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {32, 64};
    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor in0 = torch::randn({in_shapes[i]});
      auto in0_hpu = in0.to(torch::kHPU);

      auto in1 = in0.as_strided({4, 4}, {1, 4}, 0);
      in1 = torch::gelu_(in1);

      // graph -> strided_view + gelu_ + strided_insert
      auto in1_hpu = in0_hpu.as_strided({4, 4}, {1, 4}, 0);
      in1_hpu = torch::gelu_(in1_hpu);

      EXPECT_EQ(allclose(in1, in1_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, EqWithCast) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      auto input = torch::arange({in_shapes[i]}, torch::dtype(torch::kInt32))
                       .to(torch::dtype(torch::kFloat));
      auto input_hpu = input.to(torch::kHPU);

      auto output = torch::empty(0, torch::dtype(torch::kBool));
      auto output_hpu = output.to(torch::kHPU);

      torch::eq_outf(input, 1, output);
      torch::eq_outf(input_hpu, 1, output_hpu);

      EXPECT_EQ(equal(output, output_hpu.cpu()), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, EqWithCastView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {16, 32};
    for (auto i = 0; i < in_shapes.size(); i++) {
      auto input = torch::arange({in_shapes[i]}, torch::dtype(torch::kInt32))
                       .to(torch::dtype(torch::kFloat));
      auto input_hpu = input.to(torch::kHPU);

      auto input_view = input.as_strided({4, 4}, {1, 4}, 0);
      auto input_view_hpu = input_hpu.as_strided({4, 4}, {1, 4}, 0);

      auto output = torch::empty(0, torch::dtype(torch::kBool));
      auto output_hpu = output.to(torch::kHPU);

      torch::eq_outf(input_view, 0, output);
      torch::eq_outf(input_view_hpu, 0, output_hpu);

      EXPECT_EQ(equal(output, output_hpu.cpu()), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, Div) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {0, 0};
    for (auto i = 0; i < in_shapes.size(); i++) {
      auto input = torch::empty({in_shapes[i]});
      auto input_hpu = input.to(torch::kHPU);

      auto output = torch::empty(0);
      auto output_hpu = output.to(torch::kHPU);

      torch::div_outf(input, 2, output);
      torch::div_outf(input_hpu, 2, output_hpu);

      EXPECT_EQ(allclose(output, output_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, DISABLED_DivView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {256, 512};
    for (auto i = 0; i < in_shapes.size(); i++) {
      auto input =
          torch::randn({in_shapes[i], 2048}).to(torch::dtype(torch::kBFloat16));
      auto input_hpu = input.to(torch::kHPU);

      auto output = torch::empty(0, torch::dtype(torch::kBFloat16));
      auto output_hpu = output.to(torch::kHPU);

      auto input_view = input.as_strided({256, 2048, 7, 7}, {2048, 1, 0, 0}, 0);
      torch::div_outf(input_view, 2, output);

      auto input_view_hpu =
          input_hpu.as_strided({256, 2048, 7, 7}, {2048, 1, 0, 0}, 0);
      torch::div_outf(input_view_hpu, 2, output_hpu);

      EXPECT_EQ(allclose(output, output_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

// Test where strided_view with strides on FCD is replaced
// by strided_view with contiguous strides on FCD and permute op
TEST_F(ShapeAgnosticTest, StridedPermute) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    std::vector<int> in_shapes = {32, 64};
    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor in0 =
          torch::arange(in_shapes[i]).to(torch::dtype(torch::kFloat32));
      auto in0_hpu = in0.to(torch::kHPU);

      // strided on fcd i.e dim 'x'
      auto in1 = in0.as_strided({3, 4}, {1, 3}, 1);
      in1.add_(-1.0);

      // graph -> strided_view non-contiguous + add_ + strided_insert
      // after permute pass
      // graph -> strided_view contiguous + permute + add_ + strided_insert
      auto in1_hpu = in0_hpu.as_strided({3, 4}, {1, 3}, 1);
      in1_hpu.add_(-1.0);

      EXPECT_EQ(allclose(in1, in1_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

// Test where strided_view with strides on FCD is replaced
// by strided_view with contiguous strides on FCD and permute op
TEST_F(ShapeAgnosticTest, StridedPermute2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  std::vector<std::vector<int64_t>> in_shapes{
      {1024, 1024}, {64, 16, 128, 64}, {64, 128, 16, 64}, {64, 16, 64, 128}};
  std::vector<std::vector<int64_t>> in_strides{
      {1, 1024},
      {131072, 64, 1024, 1},
      {131072, 64, 8192, 1},
      {131072, 64, 1, 1024}};
  if (device.type() == synDeviceGaudi2) {
    for (auto i = 0; i < in_shapes.size(); i++) {
      int64_t total_tensor_size = std::accumulate(
          in_shapes[i].begin(),
          in_shapes[i].end(),
          1,
          std::multiplies<int64_t>());
      torch::Tensor in0 = torch::rand({total_tensor_size});
      auto in0_hpu = in0.to(torch::kHPU);

      auto in1 = in0.as_strided(in_shapes[i], in_strides[i], 0);
      auto in1_hpu = in0_hpu.as_strided(in_shapes[i], in_strides[i], 0);

      EXPECT_EQ(allclose(in1, in1_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

// Test where strided_view with strides on FCD is replaced
// by strided_view with contiguous strides on FCD and permute op
// Test with Permute order '4' with expand i.e. 0 stride on dim '1'
// Expand dim[1] - 5, org size = 6144, expand size = 6144 x 5
TEST_F(ShapeAgnosticTest, StridedPermute3) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  std::vector<std::vector<int64_t>> in_shapes{
      {16, 32, 3, 5, 4}, {16, 32, 3, 5, 4}};
  std::vector<std::vector<int64_t>> in_strides{
      {384, 1, 128, 0, 32}, {384, 1, 128, 0, 32}};
  if (device.type() == synDeviceGaudi2 || device.type() == synDeviceGaudi3) {
    for (auto i = 0; i < in_shapes.size(); i++) {
      torch::Tensor in0 = torch::rand({6144});
      auto in0_hpu = in0.to(torch::kHPU);

      auto in1 = in0.as_strided(in_shapes[i], in_strides[i], 0);
      auto in1_hpu = in0_hpu.as_strided(in_shapes[i], in_strides[i], 0);

      EXPECT_EQ(allclose(in1, in1_hpu.cpu(), 0.01, 0.01), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, Zero) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  std::vector<int64_t> shapes{{32}, {64}};
  std::vector<std::vector<int64_t>> in_shapes{{4, 4}, {4, 4}};
  std::vector<std::vector<int64_t>> in_strides{{1, 4}, {1, 4}};
  if (device.type() == synDeviceGaudi2) {
    for (auto i = 0; i < shapes.size(); i++) {
      torch::Tensor input =
          torch::rand(shapes[i]).to(torch::dtype(torch::kBFloat16));
      auto input_hpu = input.to(torch::kHPU);

      // Strides along FCD
      auto input_strided = input.as_strided(in_shapes[i], in_strides[i], 4);
      auto input_strided_hpu =
          input_hpu.as_strided(in_shapes[i], in_strides[i], 4);

      input_strided.zero_();
      input_strided_hpu.zero_();

      auto result = input_strided_hpu.cpu();
      EXPECT_EQ(torch::equal(result, input_strided), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, Fill) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  std::vector<std::vector<int64_t>> shapes{{2, 3}, {4, 6}};
  if (device.type() == synDeviceGaudi2) {
    for (auto i = 0; i < shapes.size(); i++) {
      torch::Tensor input = torch::rand(shapes[i]);
      auto input_hpu = input.to(torch::kHPU);

      input.t_().fill_(10);
      input_hpu.t_().fill_(10);

      auto result = input_hpu.cpu();
      EXPECT_EQ(torch::equal(result, input), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, CatAddView) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  std::vector<int64_t> in_shapes{32, 48};
  std::vector<int64_t> offset{4, 8};
  if (device.type() == synDeviceGaudi2) {
    for (auto i = 0; i < in_shapes.size(); i++) {
      auto a = torch::randn({8, 8}).to(torch::kBFloat16);
      auto b = torch::randn({8, 8}).to(torch::kBFloat16);
      auto c = torch::randn({8, 8}).to(torch::kBFloat16);

      auto a_view = a.as_strided({in_shapes[i]}, {1}, offset[i]);
      auto b_view = b.as_strided({in_shapes[i]}, {1}, offset[i]);
      auto c_view = c.as_strided({in_shapes[i]}, {1}, offset[i]);

      auto a_hpu = a.to(torch::kHPU);
      auto b_hpu = b.to(torch::kHPU);
      auto c_hpu = c.to(torch::kHPU);

      auto a_hpu_view = a_hpu.as_strided({in_shapes[i]}, {1}, offset[i]);
      auto b_hpu_view = b_hpu.as_strided({in_shapes[i]}, {1}, offset[i]);
      auto c_hpu_view = c_hpu.as_strided({in_shapes[i]}, {1}, offset[i]);

      auto out = torch::zeros({3 * 8 * 8}).to(torch::kBFloat16);
      auto out_view = out.as_strided({3 * in_shapes[i]}, {1}, offset[i]);

      auto out_hpu = out.to(torch::kHPU);
      auto out_hpu_view =
          out_hpu.as_strided({3 * in_shapes[i]}, {1}, offset[i]);

      torch::cat_out(out_view, {a_view, b_view, c_view});
      torch::cat_out(out_hpu_view, {a_hpu_view, b_hpu_view, c_hpu_view});

      auto d = torch::randn({3 * 8 * 8}).to(torch::kBFloat16);
      auto d_view = d.as_strided({3 * in_shapes[i]}, {1}, offset[i]);

      auto d_hpu = d.to(torch::kHPU);
      auto d_hpu_view = d_hpu.as_strided({3 * in_shapes[i]}, {1}, offset[i]);

      out_view.add_(d_view);
      out_hpu_view.add_(d_hpu_view);

      EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out, 0.001, 0.001), true);
    }
  }
}

TEST_F(ShapeAgnosticTest, LayerNormForwardExecute) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto input_tensor =
        torch::randn((480), torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({10, 1, 3, 4, 4}); // nchw
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    at::Tensor weight =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tWeight = weight.to(torch::kHPU);
    at::Tensor bias =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tBias = bias.to(torch::kHPU);
    auto results =
        torch::native_layer_norm(tHabanaX, {1, 3, 4, 4}, tWeight, tBias, 0.01);

    auto input_tensor_2 =
        torch::arange(1920, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({40, 1, 3, 4, 4}); // nchw
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);
    at::Tensor weight_2 =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tWeight_2 = weight_2.to(torch::kHPU);
    at::Tensor bias_2 =
        torch::arange(48, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape({1, 3, 4, 4}); // nchw;
    torch::Tensor tBias_2 = bias_2.to(torch::kHPU);
    auto results_2 = torch::native_layer_norm(
        tHabanaX_2, {1, 3, 4, 4}, tWeight_2, tBias_2, 0.01);

    at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
    at::Tensor result_lazy_2 = (std::get<0>(results_2)).to(torch::kCPU);
    auto results_cpu = torch::native_layer_norm(
        input_tensor, {1, 3, 4, 4}, weight, bias, 0.01);
    auto results_cpu_2 = torch::native_layer_norm(
        input_tensor_2, {1, 3, 4, 4}, weight_2, bias_2, 0.01);
    at::Tensor result_cpu = std::get<0>(results_cpu);
    at::Tensor result_cpu_2 = std::get<0>(results_cpu_2);
    // cpu and hpu results not expected match
    EXPECT_FALSE(result_lazy.equal(result_cpu));
    EXPECT_FALSE(result_lazy_2.equal(result_cpu_2));
  }
}

TEST_F(ShapeAgnosticTest, LayerNormForwardExecute2) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = habana::HPURegistrar::get_device();
  if (device.type() == synDeviceGaudi2) {
    auto input_tensor = torch::randn(
        {24, 384, 1024}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
    at::Tensor weight = torch::empty(
        {1024}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw;
    torch::Tensor tWeight = weight.to(torch::kHPU);
    at::Tensor bias = torch::empty(
        {1024}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw;
    torch::Tensor tBias = bias.to(torch::kHPU);
    auto results =
        torch::native_layer_norm(tHabanaX, {1024}, tWeight, tBias, 0.01);

    auto input_tensor_2 = torch::randn(
        {36, 576, 1024}, torch::dtype(torch::kFloat).requires_grad(false));
    torch::Tensor tHabanaX_2 = input_tensor_2.to(torch::kHPU);
    at::Tensor weight_2 = torch::empty(
        {1024}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw;
    torch::Tensor tWeight_2 = weight_2.to(torch::kHPU);
    at::Tensor bias_2 = torch::empty(
        {1024}, torch::dtype(torch::kFloat).requires_grad(false)); // nchw;
    torch::Tensor tBias_2 = bias_2.to(torch::kHPU);
    auto results_2 =
        torch::native_layer_norm(tHabanaX_2, {1024}, tWeight_2, tBias_2, 0.01);

    at::Tensor result_lazy = (std::get<0>(results)).to(torch::kCPU);
    at::Tensor result_lazy_2 = (std::get<0>(results_2)).to(torch::kCPU);
    auto results_cpu =
        torch::native_layer_norm(input_tensor, {1024}, weight, bias, 0.01);
    auto results_cpu_2 = torch::native_layer_norm(
        input_tensor_2, {1024}, weight_2, bias_2, 0.01);
    at::Tensor result_cpu = std::get<0>(results_cpu);
    at::Tensor result_cpu_2 = std::get<0>(results_cpu_2);
    // cpu and hpu results not expected match
    EXPECT_FALSE(result_lazy.equal(result_cpu));
    EXPECT_FALSE(result_lazy_2.equal(result_cpu_2));
  }
}