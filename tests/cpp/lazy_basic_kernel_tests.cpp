/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */
#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"
using namespace habana_lazy;
using namespace at;

class LazyBasicKernelTest : public habana_lazy_test::LazyTest {};

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
  // auto hl_result =
  // std::make_shared<HbLazyTensor>(SyncAndGetHbLazyTensor(hD));
  // std::vector<HbLazyTensor> tensors = {*hl_result};
  // HbLazyTensor::SyncTensorsGraph(&tensors);
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

TEST_F(LazyBasicKernelTest, ViewCopy2) {
  auto A = torch::randn({4});
  auto hA = A.to(torch::kHPU);

  auto B = torch::randn({4});
  auto hB = B.to(torch::kHPU);

  A.view(-1).add_(1.0);
  A.view(-1).mul_(2.0);
  A.copy_(B.view(-1));

  hA.view(-1).add_(1.0);
  hA.view(-1).mul_(2.0);
  hA.copy_(hB.view(-1));

  EXPECT_EQ(allclose(A, hA.cpu(), 0.001, 0.001), true);
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
  auto hB = torch::as_strided(hA, sizes, strides, offset);
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
  // Slice H2D flow only supports max 5dims : SW-153474
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE, false, 1);
  torch::Tensor A = torch::randn({2, 4, 3, 5, 6, 7})
                        .contiguous(c10::MemoryFormat::Contiguous);
  auto hA = A.to(torch::kHPU);
  auto B = torch::slice(A, 1, 1, -1, 1);
  auto hB = torch::slice(hA, 1, 1, -1, 1);
  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(B, hB.cpu()), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE);
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
TEST_F(LazyBasicKernelTest, DISABLED_asStridedOnChlastInput) {
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
TEST_F(LazyBasicKernelTest, DISABLED_asStridedOnChlastOutput) {
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

TEST_F(LazyBasicKernelTest, allreducewithcontroledge2) {
  torch::Tensor A = torch::randn({4});
  auto hA = A.to(torch::kHPU);
  A.fill_(0.0);
  auto b = torch::relu(A);
  auto v1 = A.view(-1);
  auto v2 = A.view(-1);
  auto grad1 = torch::randn({4});
  auto grad2 = torch::randn({4});

  hA.fill_(0.0);
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

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(hB)};
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

TEST_F(LazyBasicKernelTest, multilevelview) {
  torch::Tensor A = torch::randn({2, 3, 4, 5});
  auto hA = A.to(torch::kHPU);
  auto B = A.view({2 * 3, 4, 5});
  auto C = B.view({2 * 3, 4 * 5});

  auto hB = hA.view({2 * 3, 4, 5});
  auto hC = hB.view({2 * 3, 4 * 5});

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, noncontiguous) {
  auto A = torch::randn({1, 2, 2, 2});
  auto hA = A.to(torch::kHPU);
  auto B = A.as_strided({1, 2, 2, 2}, {8, 1, 4, 2});

  auto hB = hA.as_strided({1, 2, 2, 2}, {8, 1, 4, 2});
  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, unsqeezeTest) {
  auto x = torch::randn({4});
  auto hx = x.to(torch::kHPU);

  auto B = torch::unsqueeze(x, 1);
  auto hB = torch::unsqueeze(hx, 1);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, unsqeezeCmptOpTest) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  auto x = torch::randn({4});
  auto hx = x.to(torch::kHPU);

  auto B = torch::unsqueeze(x, 1);
  auto hB = torch::unsqueeze(hx, 1);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyBasicKernelTest, simultaneousinplaceoutOpTest) {
  auto a = torch::randn({2, 3});
  auto ha = a.to(torch::kHPU);

  auto b = torch::randn({3, 2});
  auto hb = b.to(torch::kHPU);

  auto a_transpose = a.transpose(0, 1);
  torch::ge_outf(a_transpose, b, a_transpose);

  // hpu
  auto ha_transpose = ha.transpose(0, 1);
  torch::ge_outf(ha_transpose, hb, ha_transpose);

  EXPECT_EQ(allclose(a_transpose, ha_transpose.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, stridedviewoutTest) {
  auto bucket = torch::randn({64});
  auto hbucket = bucket.to(torch::kHPU);

  auto gv1 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  auto gv2 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  auto gv3 = bucket.as_strided({32}, {1}, 32);

  auto g1 = torch::randn({2, 2, 2, 2});
  auto g2 = torch::randn({2, 2, 2, 2});
  auto g3 = torch::randn({32});

  gv1.mul_(g1);
  gv2.mul_(g2);
  gv3.mul_(g3);

  // hpu
  auto hgv1 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  auto hgv2 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  auto hgv3 = hbucket.as_strided({32}, {1}, 32);

  auto hg1 = g1.to(torch::kHPU);
  auto hg2 = g2.to(torch::kHPU);
  auto hg3 = g3.to(torch::kHPU);

  hgv1.mul_(hg1);
  hgv2.mul_(hg2);
  hgv3.mul_(hg3);

  HbLazyTensorViews::StepMarkerAllReduce({hbucket});

  EXPECT_EQ(allclose(hgv1.cpu(), gv1, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv2.cpu(), gv2, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv3.cpu(), gv3, 0.001, 0.001), true);

  // optimizer
  auto out = torch::mul(gv3, 0.1);
  auto hout = torch::mul(hgv3, 0.1);

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(hout.cpu(), out, 0.001, 0.001), true);

  // cache hit case
  bucket = torch::randn({64});
  hbucket = bucket.to(torch::kHPU);
  gv1 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  gv2 = bucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  gv3 = bucket.as_strided({32}, {1}, 32);

  g1 = torch::randn({2, 2, 2, 2});
  g2 = torch::randn({2, 2, 2, 2});
  g3 = torch::randn({32});

  gv1.mul_(g1);
  gv2.mul_(g2);
  gv3.mul_(g3);

  // hpu
  hgv1 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 0);
  hgv2 = hbucket.as_strided({2, 2, 2, 2}, {8, 4, 2, 1}, 16);
  hgv3 = hbucket.as_strided({32}, {1}, 32);

  hg1 = g1.to(torch::kHPU);
  hg2 = g2.to(torch::kHPU);
  hg3 = g3.to(torch::kHPU);

  hgv1.mul_(hg1);
  hgv2.mul_(hg2);
  hgv3.mul_(hg3);

  HbLazyTensorViews::StepMarkerAllReduce({hbucket});

  EXPECT_EQ(allclose(hgv1.cpu(), gv1, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv2.cpu(), gv2, 0.001, 0.001), true);
  EXPECT_EQ(allclose(hgv3.cpu(), gv3, 0.001, 0.001), true);

  // optimizer
  out = torch::mul(gv3, 0.1);
  hout = torch::mul(hgv3, 0.1);

  HbLazyTensor::StepMarker({});
  EXPECT_EQ(allclose(hout.cpu(), out, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteResizeInplaceAddTest) {
  torch::Tensor A = torch::randn({2, 3, 4, 5});
  auto hA = A.to(torch::kHPU);

  auto hOutPerm = hA.permute({0, 2, 3, 1});
  auto hOut = hOutPerm.reshape({2, 4, 3 * 5});
  hOut.add_(2);
  auto outPerm = A.permute({0, 2, 3, 1});
  auto out = outPerm.reshape({2, 4, 3 * 5});
  out.add_(2);

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, inplaceAddPermuteResizeInplaceAddTest) {
  torch::Tensor A = torch::randn({2, 3, 4, 5});
  auto hA = A.to(torch::kHPU);

  hA.add_(1);
  auto hOutPerm = hA.permute({0, 2, 3, 1});
  auto hOut = hOutPerm.reshape({2, 4, 3 * 5});
  hOut.add_(2);

  A.add_(1);
  auto outPerm = A.permute({0, 2, 3, 1});
  auto out = outPerm.reshape({2, 4, 3 * 5});
  out.add_(2);

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, permuteResizeTest) {
  torch::Tensor A = torch::randn({24, 16, 384, 64});
  auto hA = A.to(torch::kHPU);

  auto hOutPerm = hA.permute({0, 2, 1, 3}); // 24, 384, 16, 64
  auto hOut = hOutPerm.reshape({24, 384, 1024});

  auto outPerm = A.permute({0, 2, 1, 3});
  auto out = outPerm.reshape({24, 384, 1024});

  auto hOut_cpu = hOut.cpu();
  EXPECT_EQ(allclose(out, hOut_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, unsqeezeExpandTest) {
  auto x = torch::randn({});
  auto hx = x.to(torch::kHPU);

  auto B = torch::unsqueeze(x, -1);
  auto hB = torch::unsqueeze(hx, -1);

  auto C = B.expand(1);
  auto hC = hB.expand(1);

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, aliasTest) {
  auto x = torch::randn({1, 2});
  auto hx = x.to(torch::kHPU);

  auto B = at::alias(x);
  auto hB = at::alias(hx);

  auto C = B.reshape({2, 1});
  auto hC = hB.reshape({2, 1});

  auto C1 = at::alias(C);
  auto hC1 = at::alias(hC);

  auto C2 = C1.reshape({2, 1});
  auto hC2 = hC1.reshape({2, 1});

  EXPECT_EQ(allclose(C2, hC2.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, aliasTest1) {
  auto x = torch::randn({1, 2});
  auto hx = x.to(torch::kHPU);

  auto B = at::alias(x);
  auto hB = at::alias(hx);

  auto C = B.reshape({2, 1});
  auto hC = hB.reshape({2, 1});

  auto C1 = C.expand({2, 1});
  auto hC1 = hC.expand({2, 1});

  C1 = at::alias(C1);
  hC1 = at::alias(hC1);

  auto C2 = C1.reshape({2, 1});
  auto hC2 = hC1.reshape({2, 1});

  EXPECT_EQ(allclose(C2, hC2.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, aliasTest2) {
  auto x = torch::randn({});
  auto hx = x.to(torch::kHPU);

  auto B = at::alias(x);
  auto hB = at::alias(hx);

  auto C = B.reshape({1});
  auto hC = hB.reshape({1});

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, aliasTest3) {
  auto x = torch::randn({0});
  auto hx = x.to(torch::kHPU);

  auto B = at::alias(x);
  auto hB = at::alias(hx);

  auto C = B.unsqueeze(-1);
  auto hC = hB.unsqueeze(-1);

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, aliasTest4) {
  auto x = torch::randn({});
  auto hx = x.to(torch::kHPU);

  auto B = at::alias(x);
  auto hB = at::alias(hx);

  auto C = B.unsqueeze(-1);
  auto hC = hB.unsqueeze(-1);

  EXPECT_EQ(allclose(C, hC.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, multilevelview_1) {
  torch::Tensor t1 = torch::randn({1});
  torch::Tensor t2 = torch::randn({1});
  auto ht1 = t1.to(torch::kHPU);
  auto ht2 = t2.to(torch::kHPU);
  auto t3 = t1.new_full({2, 1}, -100);
  auto b1 = torch::slice(t3, 0, 0, 1, 1);
  b1.copy_(t1);
  auto b2 = torch::slice(t3, 0, 1, 2, 1);
  b2.copy_(t2);
  auto t4 = t3.new_full({3, 1}, -100);
  auto b3 = torch::slice(t4, 0, 0, 2, 1);
  b3.copy_(t3);

  // HPU
  torch::Tensor ht3 = ht1.new_full({2, 1}, -100);
  auto hb1 = torch::slice(ht3, 0, 0, 1, 1);
  hb1.copy_(ht1);
  auto hb2 = torch::slice(ht3, 0, 1, 2, 1);
  hb2.copy_(ht2);

  torch::Tensor ht4 = ht3.new_full({3, 1}, -100);
  auto hb3 = torch::slice(ht4, 0, 0, 2, 1);
  hb3.copy_(ht3);
  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(t4, ht4.cpu(), 0.001, 0.001), true);
}

// This test is same as multilevelview_1, but uses slice_insert
TEST_F(LazyBasicKernelTest, multilevelview_2) {
  torch::Tensor t1 = torch::randn({1, 1});
  torch::Tensor t2 = torch::randn({1, 1});
  auto ht1 = t1.to(torch::kHPU);
  auto ht2 = t2.to(torch::kHPU);
  auto t3 = t1.new_full({2, 1}, -100);
  auto b1 = torch::slice(t3, 0, 0, 1, 1);
  b1.copy_(t1);
  auto b2 = torch::slice(t3, 0, 1, 2, 1);
  b2.copy_(t2);
  auto t4 = t3.new_full({3, 1}, -100);
  auto b3 = torch::slice(t4, 0, 0, 2, 1);
  b3.copy_(t3);

  // HPU
  torch::Tensor ht3 = ht1.new_full({2, 1}, -100);
  auto hb1 = torch::slice(ht3, 0, 0, 1, 1);
  hb1.copy_(ht1);
  auto hb2 = torch::slice(ht3, 0, 1, 2, 1);
  hb2.copy_(ht2);

  torch::Tensor ht4 = ht3.new_full({3, 1}, -100);
  auto hb3 = torch::slice(ht4, 0, 0, 2, 1);
  hb3.copy_(ht3);
  HbLazyTensor::StepMarker({});

  EXPECT_EQ(allclose(t4, ht4.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, d2d_broadcast) {
  auto a = torch::randn({2, 2});

  auto ha = a.to(torch::kHPU);
  auto b = torch::tensor({1.0});

  auto hb = b.to(torch::kHPU);

  a.copy_(b);
  ha.copy_(hb);

  EXPECT_EQ(allclose(a, ha.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, viewinsert_broadcast) {
  auto a = torch::randn({2, 2});

  auto ha = a.to(torch::kHPU);
  auto b = torch::tensor({1.0});

  auto hb = b.to(torch::kHPU);

  a.view(-1).copy_(b);
  ha.view(-1).copy_(hb);

  EXPECT_EQ(allclose(a, ha.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, gather_neg_dim) {
  torch::Tensor inp = torch::randn({2});
  auto indx = torch::randint(0, 2, {2}, torch::kInt64);
  auto hinp = inp.to(torch::kHPU);
  auto hindx = indx.to(torch::kHPU);
  auto cpuout = torch::gather(inp, -1, indx, 0);
  auto hpuout = torch::gather(hinp, -1, hindx, 0);
  EXPECT_EQ(allclose(cpuout, hpuout.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyBasicKernelTest, DISABLED_noncontigD2H_nonblocking) {
  torch::Tensor A = torch::randn({2, 2});
  auto hA = A.to(torch::kHPU);
  std::vector<int64_t> sz{2, 2};
  std::vector<int64_t> str{1, 2};
  c10::IntArrayRef sizes(sz.data(), sz.size());
  c10::IntArrayRef strides(str.data(), str.size());

  auto out = torch::as_strided(A, sz, str).relu();
  auto hout = torch::as_strided(hA, sz, str).relu();

  auto hout_cpu =
      torch::empty_like(out).to(torch::kCPU).pin_memory(torch::kHPU);
  HbLazyTensor::StepMarker({}, nullptr, {}, true);
  hout_cpu.copy_(hout, true);

  auto fut = std::async(std::launch::async, []() {
    HbLazyTensor::StepMarkerFinish();
    habana::HPURegistrar::synchronize_device();
  });
  fut.get();
  EXPECT_EQ(allclose(out, hout_cpu, 0.001, 0.001), true);
}
