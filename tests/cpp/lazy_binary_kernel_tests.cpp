/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
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

#include <cstdlib>

using namespace habana_lazy;
using namespace at;

class LazyBinaryKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyBinaryKernelTest, LazyDoATest) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor hC = C.to(torch::kHPU);
  torch::Tensor I = torch::add(hA, hB, 2.3);
  torch::Tensor out = torch::add(hC, I, 2.3);

  torch::Tensor I_cpu = torch::add(A, B, 2.3);
  torch::Tensor out_cpu = torch::add(C, I_cpu, 2.3);
  torch::Tensor out_h = out.to(torch::kCPU);

  EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, UnifiedFlowA) {
  std::vector<int> in_sizes{2, 2, 4, 6, 8, 10, 2};

  for (int i = 0; i < in_sizes.size(); i++) {
    int dyn_dim = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({2, dyn_dim}, torch::requires_grad(false));
    torch::Tensor B = torch::randn({2, dyn_dim}, torch::requires_grad(false));
    torch::Tensor C = torch::randn({2, dyn_dim}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor I = torch::add(hA, hB, 2.3);
    torch::Tensor out = torch::add(hC, I, 2.3);

    torch::Tensor I_cpu = torch::add(A, B, 2.3);
    torch::Tensor out_cpu = torch::add(C, I_cpu, 2.3);
    torch::Tensor out_h = out.to(torch::kCPU);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);

    PT_TEST_DEBUG("PTI_DBG :: TEST ", i, "  ========\n");
  }
}

TEST_F(LazyBinaryKernelTest, UnifiedFlowB) {
  std::vector<int> in_sizes{2, 2};

  for (int i = 0; i < in_sizes.size(); i++) {
    int dyn_dim = in_sizes[i];
    PT_TEST_DEBUG("\nPTI_DBG :: TEST ", i, "  --------\n");
    torch::Tensor A = torch::randn({2, dyn_dim}, torch::requires_grad(false));
    torch::Tensor B = torch::randn({2, dyn_dim}, torch::requires_grad(false));
    torch::Tensor C = torch::randn({2, dyn_dim}, torch::requires_grad(false));
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor hB = B.to(torch::kHPU);
    torch::Tensor hC = C.to(torch::kHPU);
    torch::Tensor I = torch::add(hA, hB, 2.3);
    torch::Tensor out = torch::add(hC, I, 2.3);

    torch::Tensor I_cpu = torch::add(A, B, 2.3);
    torch::Tensor out_cpu = torch::add(C, I_cpu, 2.3);
    torch::Tensor out_h = out.to(torch::kCPU);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);

    PT_TEST_DEBUG("PTI_DBG :: TEST ", i, "  ========\n");
  }
}

TEST_F(LazyBinaryKernelTest, AddScalarTest) {
  // test case for result = add(tensor, scalar, alpha)
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  Scalar B = 222.0;
  Scalar alpha = 111.0;

  {
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor out_h = torch::add(hA, B, alpha).to(torch::kCPU);
    torch::Tensor out_cpu = torch::add(A, B, alpha);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
  }
  {
    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor out_h = torch::add(hA, alpha, B).to(torch::kCPU);
    torch::Tensor out_cpu = torch::add(A, B, alpha);

    EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
  }
}

TEST_F(LazyBinaryKernelTest, SubScalarTest) {
  // test case for result = sub(tensor, scalar, alpha)
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  Scalar B = 2.0;
  Scalar alpha = 1.0;

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor out_h = torch::sub(hA, B, alpha).to(torch::kCPU);
  torch::Tensor out_cpu = torch::sub(A, B, alpha);

  EXPECT_EQ(allclose(out_h, out_cpu, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, AddInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::randn({2, 3});
  torch::Tensor B = torch::randn({2, 3});
  torch::Tensor C = torch::randn({2, 3});

  auto hA = A.to(torch::kHPU);
  A = A.add_(B);
  auto exp = torch::mul(A, C);

  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);
  hA = hA.add_(hB);
  auto result = torch::mul(hA, hC);

  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor out = result.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, LazyRsubscalarTest) {
  torch::Tensor input = torch::ones({10, 10});

  auto hinput = input.to(torch::kHPU);
  auto hrsub = torch::rsub(hinput, 8, 2);
  Tensor hout = hrsub.to(kCPU);

  auto cout = torch::rsub(input, 8, 2);
  EXPECT_EQ(allclose(hout, cout, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, DivTensorTestWithDivByZero) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  // Make sure some elements of B are zero
  int64_t noOfElement = 1;
  size_t index[dimentions.size()];
  for (unsigned int i = 0; i < dimentions.size(); ++i) {
    noOfElement *= dimentions.at(i);
  }

  int64_t noOfZeros = std::rand() % noOfElement;
  for (int64_t i = 0; i < noOfZeros; ++i) {
    for (unsigned dim = 0; dim < dimentions.size(); ++dim) {
      index[dim] = std::rand() % (dimentions.at(dim) - 1);
    }
    B[index[0]][index[1]][index[2]] = 0.0;
  }

  // Compute expected output
  auto expected = torch::div(A, B);

  // Compute actual output
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto result = torch::div(hA, hB);
  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  Tensor generated = result.to(kCPU);

  // Compare
  EXPECT_EQ(allclose(generated, expected, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, DivTensorTestByNonZero) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  // Make sure no element of B is zero
  int64_t index[dimentions.size()];
  for (index[0] = 0; index[0] < dimentions[0]; ++index[0]) {
    for (index[1] = 0; index[1] < dimentions[1]; ++index[1]) {
      for (index[2] = 0; index[2] < dimentions[2]; ++index[2]) {
        if (std::numeric_limits<float>::epsilon() >=
            abs(0.0 - B[index[0]][index[1]][index[2]]).item<float>()) {
          B[index[0]][index[1]][index[2]] = 1.0;
        } // if(std::numeric_limits<float>::epsilon()
      }
    } // for(index[1]=0;index[1]
  }

  auto expected = torch::div(A, B);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto result = torch::div(hA, hB);
  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  Tensor generated = result.to(kCPU);

  EXPECT_EQ(allclose(generated, expected, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, DivTensorByScalar) {
  auto a = torch::ones({2, 3, 4});
  auto b = torch::div(a, 2);
  auto c = torch::div(b, 3);
  auto d = torch::div(c, 4);
  auto out = torch::div(d, 5);

  auto ha = a.to("hpu");
  auto hb = torch::div(ha, 2);
  auto hc = torch::div(hb, 3);
  auto hd = torch::div(hc, 4);
  auto hout = torch::div(hd, 5);

  EXPECT_TRUE(allclose(out, hout.to("cpu"), 0.001, 0.001));
}

TEST_F(LazyBinaryKernelTest, MulOutScalar) {
  torch::Tensor input1 = torch::randn({2, 2});
  int divFactor_ = 2;
  auto wrapped = c10::scalar_to_tensor(double(1.) / divFactor_);
  wrapped.unsafeGetTensorImpl()->set_wrapped_number(true);
  torch::Tensor out_cpu = torch::zeros_like(input1);
  torch::Tensor out_hpu = torch::zeros_like(input1).to(torch::kHPU);
  at::mul_out(out_cpu, input1, wrapped);
  at::mul_out(out_hpu, input1.to(torch::kHPU), wrapped);
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, MulOut) {
  torch::Tensor input1 = torch::randn({2, 2});
  torch::Tensor input2 = torch::randn({2, 2});
  torch::Tensor out_cpu = torch::zeros_like(input1);
  torch::Tensor out_hpu = torch::zeros_like(input1).to(torch::kHPU);
  at::mul_out(out_cpu, input1, input2);
  at::mul_out(out_hpu, input1.to(torch::kHPU), input2.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, MulOutNarrow) {
  torch::Tensor input1 =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({2, 3});
  torch::Tensor input2 =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({2, 3});

  torch::Tensor A =
      torch::arange(6, torch::dtype(torch::kFloat)).reshape({2, 3});
  torch::Tensor hA = A.to(torch::kHPU);
  Tensor out_cpu = A.as_strided({2, 3}, input2.strides(), 0);
  Tensor out_hpu = hA.as_strided({2, 3}, input2.strides(), 0);

  at::mul_out(out_cpu, input1, input2);
  at::mul_out(out_hpu, input1.to(torch::kHPU), input2.to(torch::kHPU));
  HbLazyTensor::StepMarker({});
  bool equal = A.allclose(hA.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, MulOutCast) {
  torch::Tensor input1 = torch::randn({3, 3}, torch::dtype(torch::kFloat));
  torch::Tensor input2 = torch::randn({3, 3}, torch::dtype(torch::kBFloat16));

  torch::Tensor A = torch::zeros({9}, torch::dtype(torch::kBFloat16));
  torch::Tensor hA = A.to(torch::kHPU);
  Tensor out_cpu = A.as_strided({3, 3}, {3, 1}, 0);
  Tensor out_hpu = hA.as_strided({3, 3}, {3, 1}, 0);

  at::mul_out(out_cpu, input1, input2);
  at::mul_out(out_hpu, input1.to(torch::kHPU), input2.to(torch::kHPU));
  HbLazyTensor::StepMarker({});
  bool equal = A.allclose(hA.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, Maximum) {
  torch::Tensor input1 = torch::randn({2, 2});
  torch::Tensor input2 = torch::randn({2, 2});

  torch::Tensor out_cpu = at::max(input1, input2);
  torch::Tensor out_hpu =
      at::max(input1.to(torch::kHPU), input2.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, Max2DFloat) {
  at::Tensor self = at::randn({4, 100}, at::device(at::kCPU));
  // CPU Run
  at::Tensor output_ = at::max(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::max(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));
  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, MaxOneInput1DLong) {
  at::Tensor self =
      at::randint(-50, 50, {4}, at::device(at::kCPU).dtype(at::kLong));
  auto dimValue = 0;
  // CPU Run
  at::Tensor output_ = at::max(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::max(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));
  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, MaxOneInput8DLong) {
  at::Tensor self = at::randint(
      -50,
      50,
      {4, 10, 5, 3, 4, 5, 7, 2},
      at::device(at::kCPU).dtype(at::kLong));
  auto dimValue = 0;
  // CPU Run
  at::Tensor output_ = at::max(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::max(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));
  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, MaxOneInput0DFloat) {
  auto self = at::randint(-50, 50, {}, at::device(at::kCPU));
  auto dimValue = 0;
  // CPU Run
  at::Tensor output_ = at::max(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::max(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));
  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, Minimum) {
  torch::Tensor input1 = torch::randn({2, 2});
  torch::Tensor input2 = torch::randn({2, 2});

  torch::Tensor out_cpu = at::min(input1, input2);
  torch::Tensor out_hpu =
      at::min(input1.to(torch::kHPU), input2.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, Minimum8D) {
  const std::vector<int64_t> dimentions{4, 5, 3, 2, 5, 2, 3, 6};
  torch::Tensor input1 = torch::randn(dimentions);
  torch::Tensor input2 = torch::randn(dimentions);

  torch::Tensor out_cpu = at::min(input1, input2);
  torch::Tensor out_hpu =
      at::min(input1.to(torch::kHPU), input2.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, Min2DFloat) {
  const std::vector<int64_t> dimentions{4, 5};
  torch::Tensor input1 = torch::randn(dimentions);

  torch::Tensor out_cpu = at::min(input1);
  torch::Tensor out_hpu = at::min(input1.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyBinaryKernelTest, MinOneInput0DFloat) {
  auto self = at::randint(-350, 350, {});
  auto dimValue = 0;
  // CPU Run
  at::Tensor output_ = at::min(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::min(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));

  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, MinOneInput8DLong) {
  at::Tensor self = at::randint(
      -50,
      50,
      {4, 10, 5, 3, 4, 5, 7, 2},
      at::device(at::kCPU).dtype(at::kLong));
  auto dimValue = 0;
  // CPU Run
  at::Tensor output_ = at::min(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::min(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));
  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, MinOneInput1DLong) {
  at::Tensor self =
      at::randint(-50, 50, {4}, at::device(at::kCPU).dtype(at::kLong));
  auto dimValue = 0;
  // CPU Run
  at::Tensor output_ = at::min(self);
  // Prepare HPU inputs
  at::Tensor h_self = self.to(at::device(at::kHPU));
  // HPU Run
  at::Tensor h_output_ = at::min(h_self);
  // Compare CPU vs HPU
  at::Tensor h_output__cpu = h_output_.to(at::device(at::kCPU));

  EXPECT_EQ(allclose(h_output__cpu, output_, 0, 0, true), true);
}

TEST_F(LazyBinaryKernelTest, DivOut) {
  auto a = torch::randn({2, 3, 4});
  auto b = torch::randn({2, 3, 4});
  auto out = torch::empty_like(a);
  out = torch::div_out(out, a, b);

  auto ha = a.to("hpu");
  auto hb = b.to("hpu");
  auto hout = torch::empty_like(ha);
  hout = torch::div_out(hout, ha, hb);

  EXPECT_TRUE(allclose(out, hout.to("cpu"), 0.001, 0.001));
}

TEST_F(LazyBinaryKernelTest, TypePromotion1) {
  auto typetest = [](at::Tensor (*op)(const at::Tensor&, const at::Tensor&),
                     c10::ScalarType dtype1,
                     c10::ScalarType dtype2,
                     c10::IntArrayRef size) {
    auto a = torch::randn(size).to(dtype1);
    auto b = torch::randn(size).to(dtype2);
    auto out = op(a, b);

    auto ha = a.to("hpu");
    auto hb = b.to("hpu");
    auto hout = op(ha, hb);
    EXPECT_TRUE(allclose(out, hout.to("cpu")));
  };
  typetest(&torch::div, torch::kFloat, torch::kByte, {3, 3});
  typetest(&torch::div, torch::kByte, torch::kFloat, {3, 3});
  typetest(&torch::mul, torch::kFloat, torch::kLong, {2, 3});
  typetest(&torch::mul, torch::kLong, torch::kFloat, {2, 4});
  typetest(&torch::mul, torch::kInt8, torch::kInt, {3, 4});
}

TEST_F(LazyBinaryKernelTest, TypePromotion2) {
  auto typetest =
      [](at::Tensor (*op)(const at::Tensor&, const at::Tensor&, const Scalar&),
         c10::ScalarType dtype1,
         c10::ScalarType dtype2,
         c10::IntArrayRef size) {
        auto a = torch::randn(size).to(dtype1);
        auto b = torch::randn(size).to(dtype2);
        auto out = op(a, b, 1);

        auto ha = a.to("hpu");
        auto hb = b.to("hpu");
        auto hout = op(ha, hb, 1);
        EXPECT_TRUE(allclose(out, hout.to("cpu")));
      };
  typetest(&torch::sub, torch::kFloat, torch::kLong, {3, 4});
  typetest(&torch::add, torch::kLong, torch::kFloat, {3, 4});
}

TEST_F(LazyBinaryKernelTest, MulScalarTest) {
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  Scalar s = 3.27;

  auto expected = torch::mul(A, s);

  auto hA = A.to(torch::kHPU);

  auto result = torch::mul(hA, s);
  std::vector<HbLazyTensor> tensors = {SyncAndGetHbLazyTensor(result)};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  Tensor generated = result.to(kCPU);

  EXPECT_EQ(allclose(generated, expected, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, AddSame) {
  auto b = torch::randn({2, 3, 4});
  auto a = torch::relu(b);
  auto c = torch::add(a, a, 1);
  auto out = torch::relu(c);

  auto hb = b.to("hpu");
  auto ha = torch::relu(hb);
  auto hout = torch::add(ha, ha, 1);
  auto hc = torch::relu(hout);
  EXPECT_TRUE(allclose(out, hc.to("cpu")));
}

TEST_F(LazyBinaryKernelTest, MvTest) {
  torch::Tensor m = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor v = torch::randn(3, torch::requires_grad(false));
  torch::Tensor hm = m.to(torch::kHPU);
  torch::Tensor hv = v.to(torch::kHPU);

  auto out_exp = torch::mv(m, v);
  auto hout_lazy = torch::mv(hm, hv).to(torch::kCPU);

  EXPECT_TRUE(allclose(hout_lazy, out_exp, 0.001, 0.001));
}

TEST_F(LazyBinaryKernelTest, DotTest) {
  constexpr int64_t size = 5; // Dot is defined only for 1D tensor
  torch::Tensor A = torch::randn(size, torch::requires_grad(false));
  torch::Tensor B = torch::randn(size, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor hOut = torch::dot(hA, hB).to(torch::kCPU);
  torch::Tensor cpuOut = torch::dot(A, B);

  EXPECT_EQ(allclose(hOut, cpuOut), true);
}

TEST_F(LazyBinaryKernelTest, AddcdivTest) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);
  torch::Tensor C = torch::randn(dimentions);

  Scalar alpha = 3.5;

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  auto result = at::addcdiv(hA, hB, hC, alpha);
  Tensor hOut = result.to(kCPU);

  auto cpuOut = at::addcdiv(A, B, C, alpha);

  EXPECT_EQ(allclose(hOut, cpuOut, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, AddcmulTest) {
  const std::vector<int64_t> dimentions{5, 3, 4};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);
  torch::Tensor C = torch::randn(dimentions);

  Scalar alpha = 3.5;

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hC = C.to(torch::kHPU);

  auto result = at::addcmul(hA, hB, hC, alpha);
  Tensor hOut = result.to(kCPU);

  auto cpuOut = at::addcmul(A, B, C, alpha);

  EXPECT_EQ(allclose(hOut, cpuOut, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, PersistentAddSame) {
  auto a = torch::randn({2, 3, 4});
  auto b = torch::add(a, a, 1);
  auto out = torch::relu(b);

  auto ha = a.to("hpu");
  auto hout = torch::add(ha, ha, 1);
  auto hb = torch::relu(hout);
  EXPECT_TRUE(allclose(out, hb.to("cpu")));
}

TEST_F(LazyBinaryKernelTest, Pow) {
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  Tensor expected = torch::pow(A, B);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  auto result = torch::pow(hA, hB);

  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, PowInplace) {
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  A.pow_(B);
  hA.pow_(hB);
  Tensor generated = hA.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT
  EXPECT_TRUE(at::allclose(A, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, PowTensorScalar) {
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  Scalar s = 3.27;

  Tensor expected = torch::pow(A, s);

  auto hA = A.to(torch::kHPU);

  auto result = torch::pow(hA, s);

  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, PowTensorScalarInplace) {
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  Scalar s = 3.27;

  auto hA = A.to(torch::kHPU);

  A.pow_(s);

  hA.pow_(s);

  Tensor generated = hA.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(A, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, PowScalarTensor) {
  const std::vector<int64_t> dimentions{4, 5, 3};

  torch::Tensor A = torch::randn(dimentions);
  Scalar s = 3;

  Tensor expected = torch::pow(s, A);

  auto hA = A.to(torch::kHPU);
  auto result = torch::pow(s, hA);

  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderTensorTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt16));
  torch::Tensor B = torch::tensor({4, 2}, torch::dtype(torch::kInt16));

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  Tensor expected = torch::remainder(A, B);
  auto result = torch::remainder(hA, hB);
  Tensor generated = result.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderScalarTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  Scalar B = 2;

  Tensor expected = torch::remainder(A, B);
  auto result = torch::remainder(hA, B);
  Tensor generated = result.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderTensor0DTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor B = torch::tensor(2);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  Tensor expected = torch::remainder(A, B);
  auto result = torch::remainder(hA, hB);
  Tensor generated = result.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderScalar0DTest) {
  torch::Tensor A = torch::tensor(3);

  auto hA = A.to(torch::kHPU);
  Scalar B = 2;

  Tensor expected = torch::remainder(A, B);
  auto result = torch::remainder(hA, B);
  Tensor generated = result.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderTensorOutTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor B = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor out = torch::tensor({4, 2}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hOut = out.to(torch::kHPU);

  torch::remainder_outf(A, B, out);
  torch::remainder_outf(hA, hB, hOut);
  Tensor generated = hOut.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(out, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderScalarOutTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  Scalar B = 2;
  torch::Tensor out = torch::tensor({4, 2}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  auto hOut = out.to(torch::kHPU);

  torch::remainder_outf(A, B, out);
  torch::remainder_outf(hA, B, hOut);
  Tensor generated = hOut.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(out, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderTensorOut0dTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor B = torch::tensor(3);
  torch::Tensor out = torch::tensor({4, 2}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hOut = out.to(torch::kHPU);

  torch::remainder_outf(A, B, out);
  torch::remainder_outf(hA, hB, hOut);
  Tensor generated = hOut.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(out, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderTensorResizeOutTest) {
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor B = torch::tensor(3);
  torch::Tensor out = torch::empty({1}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto hOut = out.to(torch::kHPU);

  torch::remainder_outf(A, B, out);
  torch::remainder_outf(hA, hB, hOut);
  Tensor generated = hOut.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(out, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderScalarResizeOutTest) {
  torch::Tensor A = torch::tensor({4, 2, 6}, torch::dtype(torch::kInt32));
  Scalar B = 3;
  torch::Tensor out = torch::empty({1}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  auto hOut = out.to(torch::kHPU);

  torch::remainder_outf(A, B, out);
  torch::remainder_outf(hA, B, hOut);
  Tensor generated = hOut.to(kCPU);

  double rtol = 1e-05; // NOLINT
  double atol = 1e-08; // NOLINT

  EXPECT_TRUE(at::allclose(out, generated, rtol, atol, true));
}

TEST_F(LazyBinaryKernelTest, RemainderTensorInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor B = torch::tensor({4, 2}, torch::dtype(torch::kInt32));

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  A.remainder_(B);
  auto exp = A;

  hA.remainder_(hB);
  Tensor out = hA.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, RemainderScalarInplaceTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  Scalar B = 2;

  auto hA = A.to(torch::kHPU);

  A.remainder_(B);
  auto exp = A;

  hA.remainder_(B);
  Tensor out = hA.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, RemainderTensorInplace0DTest) {
  // Inplace op as output node is not supported yet.
  torch::Tensor A = torch::tensor({4, 2}, torch::dtype(torch::kInt32));
  torch::Tensor B = torch::tensor(3);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  A.remainder_(B);
  auto exp = A;

  hA.remainder_(hB);
  Tensor out = hA.to(kCPU);

  EXPECT_EQ(allclose(out, exp, 0.001, 0.001), true);
}

TEST_F(LazyBinaryKernelTest, MaxDim8DimDimNe7Keepdim) {
  const std::vector<int64_t> dimentions{4, 5, 3, 4, 5, 2, 3, 2};
  auto dim = -7;
  auto keepdim = true;
  torch::Tensor input1 = torch::randn(dimentions);
  at::Tensor h_input1 = input1.to(at::device(at::kHPU));

  auto out_cpu = at::max(input1, dim, keepdim);
  auto out_hpu = at::max(h_input1, dim, keepdim);
  EXPECT_TRUE(
      allclose(std::get<0>(out_hpu).to(at::kCPU), std::get<0>(out_cpu)) &&
      allclose(std::get<1>(out_hpu).to(at::kCPU), std::get<1>(out_cpu)));
}

TEST_F(LazyBinaryKernelTest, MaxDim8DimDim7) {
  const std::vector<int64_t> dimentions{4, 5, 3, 4, 5, 2, 3, 2};
  auto dim = 7;
  auto keepdim = false;
  torch::Tensor input1 = torch::randn(dimentions);
  at::Tensor h_input1 = input1.to(at::device(at::kHPU));

  auto out_cpu = at::max(input1, dim, keepdim);
  auto out_hpu = at::max(h_input1, dim, keepdim);
  EXPECT_TRUE(
      allclose(std::get<0>(out_hpu).to(at::kCPU), std::get<0>(out_cpu)) &&
      allclose(std::get<1>(out_hpu).to(at::kCPU), std::get<1>(out_cpu)));
}

TEST_F(LazyBinaryKernelTest, MaxDim2Dim1) {
  const std::vector<int64_t> dimentions{4, 5};
  auto dim = 1;
  auto keepdim = false;
  torch::Tensor input1 = torch::randn(dimentions);
  at::Tensor h_input1 = input1.to(at::device(at::kHPU));

  auto out_cpu = at::max(input1, dim, keepdim);
  auto out_hpu = at::max(h_input1, dim, keepdim);
  EXPECT_TRUE(
      allclose(std::get<0>(out_hpu).to(at::kCPU), std::get<0>(out_cpu)) &&
      allclose(std::get<1>(out_hpu).to(at::kCPU), std::get<1>(out_cpu)));
}

// Also validates InferOutputMeta for GUID mult_fwd_f32 and add_fwd_f32
TEST_F(LazyBinaryKernelTest, AddFwdF32) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor out_cpu = torch::add(A, B, 2.3);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out_hpu = torch::add(hA, hB, 2.3);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID cast_f32_to_bf16, mult_fwd_bf16,
// add_fwd_bf16 and cast_bf16_to_f32
TEST_F(LazyBinaryKernelTest, AddFwdBf16) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor out_cpu =
      torch::add(A.to(torch::kBFloat16), B.to(torch::kBFloat16), 2.3);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out_hpu =
      torch::add(hA.to(torch::kBFloat16), hB.to(torch::kBFloat16), 2.3);

  EXPECT_EQ(
      allclose(
          out_hpu.to(torch::kFloat).to(torch::kCPU),
          out_cpu.to(torch::kFloat),
          0.001,
          0.001),
      true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID cast_f32_to_i32, mult_fwd_i32,
// add_fwd_i32 and cast_i32_to_f32
TEST_F(LazyBinaryKernelTest, AddFwdI32withCast) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor out_cpu =
      torch::add(A.to(torch::kInt32), B.to(torch::kInt32), 3);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out_hpu =
      torch::add(hA.to(torch::kInt32), hB.to(torch::kInt32), 3);

  EXPECT_EQ(
      allclose(
          out_hpu.to(torch::kFloat).to(torch::kCPU),
          out_cpu.to(torch::kFloat),
          0.001,
          0.001),
      true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID mult_fwd_f32/bf16/i32,
// Constant_f32/bf16/i32 and add_fwd_f32/bf16/i32 with second argument as scalar
TEST_F(LazyBinaryKernelTest, AddFwdWithScalar) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto fn = [](c10::ScalarType tensor_type) {
    // test case for result = add(tensor, scalar, alpha)
    PT_TEST_DEBUG(
        "PTI_DBG: AddFwdWithScalar called for -- ", tensor_type, " ----\n");
    torch::Tensor A =
        torch::randn({2, 2}, torch::requires_grad(false)).to(tensor_type);
    Scalar B = 3.0;
    Scalar alpha = 2.0;

    torch::Tensor hA = A.to(torch::kHPU);
    torch::Tensor out_hpu = torch::add(hA, B, alpha);
    torch::Tensor out_cpu = torch::add(A, B, alpha);

    EXPECT_EQ(
        allclose(
            out_hpu.to(torch::kFloat32).to(torch::kCPU),
            out_cpu.to(torch::kFloat32),
            0.001,
            0.001),
        true);
  };

  fn(torch::kFloat32);
  fn(torch::kBFloat16);
  fn(torch::kInt32);

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID sub_fwd_f32
TEST_F(LazyBinaryKernelTest, SubFwdF32) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A = torch::randn({3, 4}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({3, 4}, torch::requires_grad(false));
  torch::Tensor out_cpu = torch::sub(A, B);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out_hpu = torch::sub(hA, hB);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID sub_fwd_f16
TEST_F(LazyBinaryKernelTest, SubFwdBf16) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  torch::Tensor A =
      torch::randn({3, 4}, torch::dtype(torch::kBFloat16).requires_grad(false));
  torch::Tensor B =
      torch::randn({3, 4}, torch::dtype(torch::kBFloat16).requires_grad(false));
  torch::Tensor out_cpu = torch::sub(A, B);

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hB = B.to(torch::kHPU);
  torch::Tensor out_hpu = torch::sub(hA, hB);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu, 0.001, 0.001), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID div_fwd_f32
TEST_F(LazyBinaryKernelTest, DivFwdF32Scalar) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto a = torch::ones({2, 3, 4});
  auto b = torch::div(a, 2);
  auto out = torch::div(b, 3);

  auto ha = a.to("hpu");
  auto hb = torch::div(ha, 2);
  auto hout = torch::div(hb, 3);

  EXPECT_TRUE(allclose(out, hout.to("cpu"), 0.001, 0.001));
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID div_fwd_f16
TEST_F(LazyBinaryKernelTest, DivFwdBf16Scalar) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  auto a = torch::ones({2, 3, 4}, torch::dtype(torch::kBFloat16));
  auto b = torch::div(a, 2);
  auto out = torch::div(b, 3);

  auto ha = a.to("hpu");
  auto hb = torch::div(ha, 2);
  auto hout = torch::div(hb, 3);

  EXPECT_TRUE(allclose(out, hout.to("cpu"), 0.001, 0.001));
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID pow_fwd_f32
TEST_F(LazyBinaryKernelTest, PowFwdF32) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  const std::vector<int64_t> dimentions{4, 5, 3};
  torch::Tensor A = torch::randn(dimentions);
  torch::Tensor B = torch::randn(dimentions);

  Tensor expected = torch::pow(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto result = torch::pow(hA, hB);
  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

// Also validates InferOutputMeta for GUID pow_fwd_f16
TEST_F(LazyBinaryKernelTest, PowFwdF16) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }
  const std::vector<int64_t> dimentions{4, 5, 3};
  torch::Tensor A = torch::ones(dimentions, torch::dtype(torch::kBFloat16));
  torch::Tensor B = torch::rand(dimentions, torch::dtype(torch::kBFloat16));

  Tensor expected = torch::pow(A, B);
  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);
  auto result = torch::pow(hA, hB);
  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

enum class OpMode {
  Normal,
  Inplace,
  Output,
};

// Also validates InferOutputMeta for GUID atan2_fwd_f32
static void TestAtan2(OpMode opMode, bool ndims) {
  if (!GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE)) {
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  }

  std::vector<int64_t> dimensions;
  if (ndims) {
    dimensions = {2, 3, 2, 3, 2, 3, 2, 3};
  } else {
    dimensions = {4, 5, 3};
  }
  torch::Tensor A = torch::rand(dimensions);
  torch::Tensor B = torch::rand(dimensions);

  auto hA = A.to(torch::kHPU);
  auto hB = B.to(torch::kHPU);

  Tensor expected;
  Tensor result;
  switch (opMode) {
    case OpMode::Normal:
      expected = torch::atan2(A, B);
      result = torch::atan2(hA, hB);
      break;
    case OpMode::Inplace:
      expected = A.atan2_(B);
      result = hA.atan2_(hB);
      break;
    case OpMode::Output:
      if (ndims) {
        // Maximum supported input/output tensor dimensions for constant_f32
        // is 5. So we can't initialize with zeros_like or anything else
        // similar.
        expected = A;
        result = hA;
      } else {
        expected = torch::zeros_like(A);
        result = torch::zeros_like(hA);
      }
      at::atan2_out(expected, A, B);
      at::atan2_out(result, hA, hB);
      break;
  }

  Tensor generated = result.to(kCPU);

  double rtol = 1e-03; // NOLINT
  double atol = 1e-03; // NOLINT

  EXPECT_TRUE(at::allclose(expected, generated, rtol, atol, true));
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyBinaryKernelTest, Atan2FwdF32) {
  TestAtan2(OpMode::Normal, false);
}

TEST_F(LazyBinaryKernelTest, Atan2FwdF32Inplace) {
  TestAtan2(OpMode::Inplace, false);
}

TEST_F(LazyBinaryKernelTest, Atan2FwdF32Out) {
  TestAtan2(OpMode::Output, false);
}

TEST_F(LazyBinaryKernelTest, Atan2FwdF32Nd) {
  TestAtan2(OpMode::Normal, true);
}

TEST_F(LazyBinaryKernelTest, Atan2FwdF32InplaceNd) {
  TestAtan2(OpMode::Inplace, true);
}

TEST_F(LazyBinaryKernelTest, Atan2FwdF32OutNd) {
  TestAtan2(OpMode::Output, true);
}
