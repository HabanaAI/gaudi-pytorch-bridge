#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
using namespace habana_lazy;
using namespace at;

class LazyBasicKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyBasicKernelTest, DISABLED_BasicThreadSafety) {
  torch::Tensor A = torch::rand({20});
  torch::Tensor hA = A.to("hpu");

  auto t = std::thread([&]() {
    torch::Tensor g = torch::ones({5});
    torch::Tensor hg = g.to("hpu");
    Tensor Out = A.narrow(0, 2, 5);
    Tensor hOut = hA.narrow(0, 2, 5);
    Out.copy_(g.view({-1}), true);
    hOut.copy_(hg.view({-1}), true);
    HbLazyTensor::StepMarker({});
  });

  auto t2 = std::thread([&]() {
    torch::Tensor hg2, g2 = torch::zeros({5});
    hg2 = g2.to("hpu");
    Tensor Out2 = A.narrow(0, 8, 5);
    Tensor hOut2 = hA.narrow(0, 8, 5);
    Out2.copy_(g2.view({-1}), true);
    hOut2.copy_(hg2.view({-1}), true);
    HbLazyTensor::StepMarker({});
  });
  t.join();
  t2.join();

  A = A.div_(2);
  hA = hA.div_(2);

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(A, hA.to("cpu"), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, DoubleCopyTest) {
  at::TensorOptions opts =
      at::TensorOptions().dtype(c10::ScalarType::Double).requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  // This should be double
  bool equal = hA_cpu.allclose(A, 0.1, 0.1);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, BasicCopyTest) {
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hA_cpu = hA.to(torch::kCPU);
  bool equal = hA_cpu.allclose(A, 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, CloneTest) {
  at::TensorOptions opts = at::TensorOptions().requires_grad(false);
  torch::Tensor A = torch::randn({50, 50}, opts);
  torch::Tensor B = torch::randn({50, 50}, opts);
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor hC = hA + hB;
  torch::Tensor hD = torch::clone(hC);
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(hD));
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  torch::Tensor hC_cpu = hC.to(torch::kCPU);
  torch::Tensor hd_cpu = hD.to(torch::kCPU);
  bool equal = hC_cpu.allclose(hd_cpu, 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBasicKernelTest, permuteTest) {
  torch::Tensor A = torch::randn({5, 6, 24, 24});
  auto hA = A.to(torch::kHPU);

  auto hOut = hA.permute({0, 2, 3, 1});
  auto out = A.permute({0, 2, 3, 1});

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteContCLTest) {
  torch::Tensor A =
      torch::randn({5, 6, 24, 24}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);

  auto hOut = hA.permute({0, 2, 3, 1});
  auto out = A.permute({0, 2, 3, 1});

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteCLTest) {
  torch::Tensor A =
      torch::randn({5, 6, 24, 24}).to(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);

  auto hOut = hA.permute({0, 2, 3, 1});
  auto out = A.permute({0, 2, 3, 1});

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteTest2) {
  torch::Tensor A = torch::randn({5, 6, 24, 24});
  auto hA = A.permute({0, 2, 3, 1}).to(torch::kHPU);
  auto out = A.permute({0, 2, 3, 1});
  auto hOut_cpu = hA.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteContCLTest2) {
  torch::Tensor A =
      torch::randn({5, 6, 24, 24}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.permute({0, 2, 3, 1}).to(torch::kHPU);
  auto out = A.permute({0, 2, 3, 1});
  auto hOut_cpu = hA.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteCLTest2) {
  torch::Tensor A =
      torch::randn({5, 6, 24, 24}).to(c10::MemoryFormat::ChannelsLast);
  auto hA = A.permute({0, 2, 3, 1}).to(torch::kHPU);
  auto out = A.permute({0, 2, 3, 1});
  auto hOut_cpu = hA.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteTest5D) {
  torch::Tensor A = torch::randn({5, 2, 6, 24, 24});
  auto hA = A.to(torch::kHPU);

  auto hOut = hA.permute({0, 2, 3, 4, 1});
  auto out = A.permute({0, 2, 3, 4, 1});

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, noncontigD2H) {
  torch::Tensor A = torch::randn({2, 2});
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{2, 2};
  std::vector<int64_t> str{1, 2};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());

  auto out = torch::as_strided(A, sz, str);
  auto hout = torch::as_strided(hA, sz, str);

  auto hout_cpu = hout.cpu();
  EXPECT_EQ(allclose(out, hout_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, noncontigD2H_test2) {
  torch::Tensor A = torch::randn({1, 2, 2});
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{2, 2};
  std::vector<int64_t> str{1, 2};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());

  auto out = torch::as_strided(A, sz, str);
  auto hout = torch::as_strided(hA, sz, str);

  auto hout_cpu = hout.cpu();
  EXPECT_EQ(allclose(out, hout_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, ViewCopy) {
  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHPU);
  Tensor Out = A.narrow(0, 2, 5);
  Tensor hOut = hA.narrow(0, 2, 5);
  torch::Tensor g = torch::ones({5});
  torch::Tensor hg = g.to(torch::kHPU);
  Out.copy_(g.view({-1}), true);
  hOut.copy_(hg.view({-1}), true);

  Tensor Out2 = A.narrow(0, 8, 5);
  Tensor hOut2 = hA.narrow(0, 8, 5);
  torch::Tensor g2 = torch::zeros({5});
  torch::Tensor hg2 = g2.to(torch::kHPU);
  Out2.copy_(g2.view({-1}), true);
  hOut2.copy_(hg2.view({-1}), true);
  HbLazyTensor::StepMarker({});
  A = A.div_(2);
  hA = hA.div_(2);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(hA.to(torch::kCPU), A), true);
}

TEST_F(LazyBasicKernelTest, NarrowInplaceOffsets) {
  torch::Tensor A = torch::randn({20});
  torch::Tensor hA = A.to(torch::kHPU);

  // cpu
  auto temp1 = A.narrow(0, 2, 5);
  auto temp2 = A.narrow(0, 7, 11);

  auto out1 = temp1.fill_(1.0);
  auto out2 = temp2.fill_(2.0);

  // hpu
  auto htemp1 = hA.narrow(0, 2, 5);
  auto htemp2 = hA.narrow(0, 7, 11);

  auto hout1 = htemp1.fill_(1.0);
  auto hout2 = htemp2.fill_(2.0);

  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(hA.cpu(), A), true);
}

TEST_F(LazyBasicKernelTest, ControlEdge) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});
  torch::Tensor F = torch::randn({2, 3});
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  auto hF = F.to(torch::kHPU);

  auto D = A.mul(B);
  auto E = C.mul(B);
  B = B.add_(F);

  auto hD = hA.mul(hB);
  auto hE = hC.mul(hB);
  hB = hB.add_(hF);
  HbLazyTensor::StepMarker({});
  Tensor out = hB.to(kCPU);

  EXPECT_EQ(allclose(out, B), true);
}
TEST_F(LazyBasicKernelTest, asStridedOnlyGraph) {
  torch::Tensor A = torch::randn({16});
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{4};
  std::vector<int64_t> str{1};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());
  int64_t offset = 0;
  auto hB = as_strided_hpu_lazy(hA, sizes, strides, offset);
  Tensor out = hB.to(kCPU);
}

TEST_F(LazyBasicKernelTest, weightsharinggraphcycle) {
  torch::Tensor A = torch::randn({16});
  torch::Tensor B = torch::randn({16});
  auto C = B.add(A);
  C.copy_(B);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto hC = hB.add(hA);
  hC.copy_(hB);

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, getTensorForScalarNoDtype) {
  auto opt = TensorOptions();
  EXPECT_EQ(opt.has_dtype(), false);
  auto tensor = get_tensor_for_scalar(0.0);
  EXPECT_EQ(tensor.scalar_type(), torch::kFloat);
}

TEST_F(LazyBasicKernelTest, SliceOnChlastInput) {
  torch::Tensor A =
      torch::randn({2, 4, 3, 5}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);
  auto B = torch::slice(A, 1, 1, -1, 1);
  auto hB = torch::slice(hA, 1, 1, -1, 1);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(B, hB.cpu()), true);
}
TEST_F(LazyBasicKernelTest, SliceOnChlast6dInput) {
  torch::Tensor A = torch::randn({2, 4, 3, 5, 6, 7})
                        .contiguous(c10::MemoryFormat::Contiguous);
  auto hA = A.to(torch::kHPU);
  auto B = torch::slice(A, 1, 1, -1, 1);
  auto hB = torch::slice(hA, 1, 1, -1, 1);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(B, hB.cpu()), true);
}
TEST_F(LazyBasicKernelTest, SelectOnChlast3dInput) {
  torch::Tensor A = torch::randn({2, 4, 3, 5, 6})
                        .contiguous(c10::MemoryFormat::ChannelsLast3d);
  auto hA = A.to(torch::kHPU);
  auto B = torch::select(A, 3, 1);
  auto hB = torch::select(hA, 3, 1);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(B, hB.cpu()), true);
}
TEST_F(LazyBasicKernelTest, SelectOnChlastInput) {
  torch::Tensor A =
      torch::randn({2, 4, 3, 5}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);
  auto B = torch::select(A, 3, 1);
  auto hB = torch::select(hA, 3, 1);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(B, hB.cpu()), true);
}
TEST_F(LazyBasicKernelTest, asStridedOnChlastInput) {
  torch::Tensor A =
      torch::randn({2, 3, 4, 5}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{2, 4};
  std::vector<int64_t> str{4, 1};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());
  int64_t offset = 0;
  auto out = torch::as_strided(A, sizes, strides, offset);
  auto hOut = torch::as_strided(hA, sizes, strides, offset);
  EXPECT_EQ(allclose(out, hOut.cpu()), true);
}
TEST_F(LazyBasicKernelTest, asStridedOnChlastOutput) {
  torch::Tensor A =
      torch::randn({2, 3, 4, 5}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{2, 4};
  std::vector<int64_t> str{4, 1};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());
  int64_t offset = 0;
  auto B = torch::relu(A);
  auto out = torch::as_strided(B, sizes, strides, offset);
  auto hB = torch::relu(hA);
  auto hOut = torch::as_strided(hB, sizes, strides, offset);
  EXPECT_EQ(allclose(out, hOut.cpu()), true);
}
TEST_F(LazyBasicKernelTest, InplaceView) {
  torch::Tensor A = torch::randn({2, 3, 4, 5});
  auto hA = A.to(torch::kHPU);
  auto B = A.view(-1);
  B.add_(0.5);
  // hpu
  auto hB = hA.view(-1);
  hB.add_(0.5);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(A, hA.cpu()), true);
}

TEST_F(LazyBasicKernelTest, allreduce) {
  torch::Tensor A = torch::randn({4});
  auto v1 = A.view(-1);
  auto v2 = A.view(-1);
  auto grad1 = torch::randn({4});
  auto grad2 = torch::randn({4});

  auto hA = A.to(torch::kHPU);
  auto hv1 = hA.view(-1);
  auto hv2 = hA.view(-1);
  auto hgrad1 = grad1.to(torch::kHPU);
  auto hgrad2 = grad2.to(torch::kHPU);

  v1.mul_(grad1);
  v2.mul_(grad2);

  hv1.mul_(hgrad1);
  hv2.mul_(hgrad2);

  EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, allreducewithcontroledge) {
  torch::Tensor A = torch::randn({4});
  auto b = torch::relu(A);
  auto v1 = A.view(-1);
  auto v2 = A.view(-1);
  auto grad1 = torch::randn({4});
  auto grad2 = torch::randn({4});

  auto hA = A.to(torch::kHPU);
  auto hB = torch::relu(hA);
  auto hv1 = hA.view(-1);
  auto hv2 = hA.view(-1);
  auto hgrad1 = grad1.to(torch::kHPU);
  auto hgrad2 = grad2.to(torch::kHPU);

  v1.mul_(grad1);
  v2.mul_(grad2);

  hv1.mul_(hgrad1);
  hv2.mul_(hgrad2);

  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, InplaceViewon3d) {
  torch::Tensor A = torch::randn({2, 3, 4, 5, 6});
  auto hA = A.to(torch::kHPU);
  auto B = A.view(-1);
  B.add_(0.5);
  // hpu
  auto hB = hA.view(-1);
  hB.add_(0.5);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(A, hA.cpu()), true);
}

TEST_F(LazyBasicKernelTest, InplaceSliceonChlast) {
  int N = 2, C = 3, H = 4, W = 5;
  torch::Tensor A =
      torch::randn({N, C, H, W}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);
  auto B = A.slice(1, 1, 3, 1);
  B.add_(0.5);

  // hpu
  auto hB = hA.slice(1, 1, 3, 1);
  hB.add_(0.5);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(A, hA.cpu()), true);
}

TEST_F(LazyBasicKernelTest, InplaceSliceonChlast3d) {
  int N = 2, C = 3, D = 4, H = 5, W = 6;
  torch::Tensor A = torch::randn({N, C, D, H, W})
                        .contiguous(c10::MemoryFormat::ChannelsLast3d);
  auto hA = A.to(torch::kHPU);
  auto B = A.slice(1, 1, 3, 1);
  B.add_(0.5);
  // hpu
  auto hB = hA.slice(1, 1, 3, 1);
  hB.add_(0.5);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(A, hA.cpu()), true);
}

TEST_F(LazyBasicKernelTest, FlattenChlast) {
  int N = 2, C = 3, D = 4, H = 5;
  torch::Tensor A =
      torch::randn({N, C, D, H}).contiguous(c10::MemoryFormat::ChannelsLast);
  auto hA = A.to(torch::kHPU);
  A = torch::flatten(A, 1);

  hA = torch::flatten(hA, 1);
  EXPECT_EQ(allclose(A, hA.cpu()), true);
}

TEST_F(LazyBasicKernelTest, d2hsync) {
  torch::Tensor A = torch::randn({3, 3});
  auto hA = A.to(torch::kHPU);
  auto B = A.as_strided({2, 2}, {1, 2}, 1);
  auto C = B.add(1.0);

  auto hB = hA.as_strided({2, 2}, {1, 2}, 1);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(hB)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto hC = hB.add(1.0);

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, viewtranspose) {
  torch::Tensor A = torch::randn({4});
  auto hA = A.to(torch::kHPU);
  auto B = A.view({2, 2});
  auto C = torch::transpose(B, 0, 1);

  auto hB = hA.view({2, 2});

  auto hC = torch::transpose(hB, 0, 1);

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}
