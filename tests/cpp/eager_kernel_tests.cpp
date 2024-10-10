/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

#include <ATen/ExpandUtils.h>
#include <gtest/gtest.h>
#include <math.h>
#include <torch/torch.h>
#include <stdexcept>
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/synapse_helpers/env_flags.h"
#include "common_functions_norm_kernel_tests.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/linear_kernels.h"
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;

class EagerKernelTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetEagerMode();
  }
  void TearDown() override {
    RestoreMode();
  }
};

class EagerKernelCacheTest : public habana_lazy_test::LazyTest {
  void SetUp() override {
    SetLazyMode(2);
  }
  void TearDown() override {
    RestoreMode();
  }
};

TEST_F(EagerKernelTest, LinspaceOutCache) {
  const int64_t constStepsValue = 11;
  torch::Scalar start = 0.0f;
  torch::Scalar end = 10.0f;
  int64_t step = constStepsValue;
  torch::Tensor out =
      torch::randn({constStepsValue}, torch::requires_grad(false));
  auto hOut = out.to(torch::kHPU);

  torch::Tensor out2 =
      torch::randn({constStepsValue}, torch::requires_grad(false));
  auto hOut2 = out2.to(torch::kHPU);

  auto h_a = torch::linspace_outf(start, end, step, hOut);
  auto h_b = torch::linspace_outf(start, end, step, hOut2);
  auto hOut_cpu = h_b.to(torch::kCPU);

  auto a = torch::linspace_outf(start, end, step, out);
  EXPECT_EQ(allclose(hOut_cpu, out), true);
}

TEST_F(EagerKernelTest, DISABLED_LinspaceOutNeToPosStep1) {
  const int64_t constStepsValue = 12; // set incorrect size
  torch::Scalar start = -100.0f;
  torch::Scalar end = 200.0f;
  int64_t step = 1;
  torch::Tensor out =
      torch::randn({constStepsValue}, torch::requires_grad(false));
  auto hOut = out.to(torch::kHPU);

  auto h_a = torch::linspace_outf(start, end, step, hOut);
  auto hOut_cpu = h_a.to(torch::kCPU);

  auto a = torch::linspace_outf(start, end, step, out);
  EXPECT_EQ(allclose(hOut_cpu, out), true);
}

TEST_F(EagerKernelTest, MinTest0D) {
  torch::Tensor A = torch::tensor(2.03);
  auto hinput = A.to(torch::kHPU);

  auto hresult = torch::min(hinput);
  auto hout = hresult.to(torch::kCPU);

  auto cpu_out = torch::min(A);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

// TODO: Add test dim from actual model's data
TEST_F(EagerKernelTest, MinTest) {
  torch::Tensor A = torch::randn({2, 3, 4, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);
  auto hOut = torch::min(hA);
  auto Out = torch::min(A);
  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(EagerKernelTest, Cumsum0D) {
  torch::Tensor A = torch::tensor(9.03);
  auto hinput = A.to(torch::kHPU);

  auto hresult = torch::cumsum(hinput, 0, torch::kFloat32);
  auto hout = hresult.to(torch::kCPU);

  auto cpu_out = torch::cumsum(A, 0, torch::kFloat32);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(EagerKernelTest, CumsumDim3AxisNe1) {
  auto A = torch::randn({2, 3, 2}, torch::requires_grad(false));

  auto hA = A.to(torch::kHPU);
  int64_t axis = -1;
  torch::Tensor cpu_out = torch::cumsum(A, axis);

  torch::Tensor hresult = torch::cumsum(hA, axis);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(EagerKernelTest, CumsumDim3Axis2) {
  torch::Tensor A = torch::randn({2, 3, 2});

  auto hA = A.to(torch::kHPU);
  int64_t axis = 2;
  torch::Tensor cpu_out = torch::cumsum(A, axis);

  torch::Tensor hresult = torch::cumsum(hA, axis);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(EagerKernelTest, CumsumDim2Axis1Int) {
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  torch::Tensor A = torch::randint(-330, 330, {2, 3}, options);

  auto hA = A.to(torch::kHPU);
  int64_t axis = 1;
  torch::Tensor cpu_out = torch::cumsum(A, axis, torch::kInt);

  torch::Tensor hresult = torch::cumsum(hA, axis, torch::kInt);
  auto hout = hresult.to(torch::kCPU);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(EagerKernelTest, ReluTest) {
  torch::Tensor tensor = torch::randn({2, 3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::relu(tHabana);
  auto out = torch::relu(tensor);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, LeakyReluTest) {
  auto A = torch::tensor({-1.0, 5.0, -1.0});
  auto ha = A.to(torch::kHPU);

  auto hA = ha.clone();
  auto cpu_out = torch::leaky_relu(A);
  auto hpu_out = torch::leaky_relu(hA);

  EXPECT_TRUE(allclose(cpu_out, hpu_out.to("cpu")));
}

TEST_F(EagerKernelTest, LeakyRelu0DTest) {
  auto A = torch::tensor(-1.0);
  auto ha = A.to(torch::kHPU);

  auto hA = ha.clone();
  auto cpu_out = torch::leaky_relu(A);
  auto hpu_out = torch::leaky_relu(hA);

  EXPECT_TRUE(allclose(cpu_out, hpu_out.to("cpu")));
}

TEST_F(EagerKernelTest, LeakyReluInplaceTest) {
  auto A = torch::tensor({-1.0, 5.0});
  auto ha = A.to(torch::kHPU);

  auto hA = ha.clone();
  torch::leaky_relu_(A);
  torch::leaky_relu_(hA);

  EXPECT_TRUE(allclose(A, hA.to("cpu")));
}

TEST_F(EagerKernelTest, LeakyReluBackwardTest) {
  const std::vector<int64_t> dimentions{2, 3};

  auto grad = torch::randn(dimentions, torch::requires_grad(false));
  auto A = torch::randn(dimentions, torch::requires_grad(false));

  auto hgrad = grad.to(torch::kHPU);
  auto hA = A.to(torch::kHPU);

  auto expectedOutput = torch::leaky_relu_backward(grad, A, 0.1, false);
  auto habanaOutput = torch::leaky_relu_backward(hgrad, hA, 0.1, false);

  EXPECT_TRUE(allclose(expectedOutput, habanaOutput.to("cpu")));
}

TEST_F(EagerKernelTest, LeakyReluBackward0DTest) {
  auto grad = torch::tensor(6.0);
  auto A = torch::tensor(-1.0);

  auto hgrad = grad.to(torch::kHPU);
  auto hA = A.to(torch::kHPU);

  auto expectedOutput = torch::leaky_relu_backward(grad, A, 0.1, false);
  auto habanaOutput = torch::leaky_relu_backward(hgrad, hA, 0.1, false);

  EXPECT_TRUE(allclose(expectedOutput, habanaOutput.to("cpu")));
}

TEST_F(EagerKernelTest, AddTest) {
  torch::Tensor tensor = torch::randn({2, 3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::add(tHabana, 4.0);
  auto out = torch::add(tensor, 4.0);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_TRUE(equal);
}

TEST_F(EagerKernelTest, MatMulTest) {
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    torch::Tensor tensor1 = torch::randn(size1);
    torch::Tensor tensor2 =
        torch::randn(size2); // torch::randn({2, 2}); d2 =1,2 tested and passing
    torch::Tensor ht1 = tensor1.to(torch::kHPU);
    torch::Tensor ht2 = tensor2.to(torch::kHPU);
    auto outHabana = torch::matmul(ht1, ht2);
    auto out = torch::matmul(tensor1, tensor2);
    bool equal;
    auto& device = habana::HPURegistrar::get_device();
    equal = out.allclose(outHabana.to(torch::kCPU), 0.001, 0.001);
    EXPECT_TRUE(equal);
  };

  // Testing all configurations supported by CPU.
  // Do not delete from this list
  matmul_test({10}, {10});
  matmul_test({2, 10}, {10});
  matmul_test({10}, {10, 2});
  matmul_test({2, 10}, {10, 2});
  matmul_test({2, 3, 4}, {4});
  matmul_test({2, 3, 4}, {2, 4, 3});
  matmul_test({12, 20, 24}, {24, 20});
  matmul_test({12, 16, 20, 24}, {12, 16, 24, 20});
  matmul_test({3}, {2, 3, 4});
  matmul_test({3, 4}, {2, 4, 3});
  matmul_test({12, 16, 20, 24}, {16, 24, 20});
  matmul_test({16, 20, 24}, {12, 16, 24, 20});
  matmul_test({10, 8, 16}, {1, 16, 12});
  matmul_test({2, 10, 8, 16}, {2, 1, 16, 12});
}

TEST_F(EagerKernelTest, WhereTest) {
  torch::Tensor x = torch::randn({2, 3});
  torch::Tensor y = torch::randn({2, 3});
  // Operator is obsoleted
  // auto out = torch::_s_where(x > 0, x, y);

  // auto hx = x.to(torch::kHPU);
  // auto hy = y.to(torch::kHPU);
  // auto outHabana = torch::_s_where(hx > 0, hx, hy);

  // auto result = outHabana.to(torch::kCPU);

  // bool equal = out.allclose(result, 0.001, 0.001);
  // EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, WhereBroadcastTest) {
  torch::Tensor cond = torch::randint(0, 2, {2, 3});
  torch::Tensor condBool = cond > 0;
  torch::Tensor x = torch::randn({2, 3});
  torch::Tensor y = torch::randn({1});

  // auto out = torch::_s_where(condBool, x, y);

  // auto hcond = condBool.to(torch::kHPU);
  // auto hx = x.to(torch::kHPU);
  // auto hy = y.to(torch::kHPU);
  // auto outHabana = torch::_s_where(hcond, hx, hy);

  // auto result = outHabana.to(torch::kCPU);

  // bool equal = out.allclose(result, 0.001, 0.001);
  // EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, IsfiniteTest) {
  auto input_tensor = torch::Tensor(torch::zeros({5}));
  input_tensor[0] = input_tensor[0] / 0.0;
  input_tensor[1] = 2.0 / 0.0;
  input_tensor[2] = -2.0 / 0.0;

  torch::Tensor cpu_out = torch::isfinite(input_tensor);

  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::isfinite(tHabanaX);
  torch::Tensor hout = outHabana.to(torch::kCPU);

  bool equal = cpu_out.equal(hout);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, IsnanTest) {
  auto input_tensor = torch::tensor({2.0, sqrt(-1.0), 1.0});

  torch::Tensor cpu_out = torch::isnan(input_tensor);
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::isnan(tHabanaX);
  torch::Tensor hout = outHabana.to(torch::kCPU);
  bool equal = cpu_out.equal(hout);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, Isnan0DTest) {
  auto input_tensor = torch::tensor(sqrt(-1.0));

  torch::Tensor cpu_out = torch::isnan(input_tensor);
  torch::Tensor tHabanaX = input_tensor.to(torch::kHPU);
  torch::Tensor outHabana = torch::isnan(tHabanaX);
  torch::Tensor hout = outHabana.to(torch::kCPU);
  bool equal = cpu_out.equal(hout);
  EXPECT_EQ(equal, true);
}

LAYER_NORM_TEST(EagerKernelTest, Forward)
LAYER_NORM_TEST(EagerKernelTest, Backward)

TEST_F(EagerKernelTest, IndexTest) {
  torch::Tensor input_cpu = torch::arange(4).reshape({2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({{0, 1}, {0, 1}})};
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  // auto tensorlist = indices.vec();
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  // auto tensorlist = indices.vec();
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  // TODO: Check index_hpu_lazy why this long cast is required
  bool equal =
      out_cpu.allclose(out_hpu.to(torch::kCPU).to(at::kLong), 0.001, 0.001);
  EXPECT_EQ(equal, true);
};

TEST_F(EagerKernelTest, BroadCastIndexTest) {
  torch::Tensor input_cpu = torch::arange(4).reshape({2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({1}), torch::tensor({0, 1})};
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  // auto tensorlist = indices.vec();
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  // auto tensorlist = indices.vec();
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  // TODO: Check index_hpu_lazy why this long cast is required
  bool equal =
      out_cpu.allclose(out_hpu.to(torch::kCPU).to(at::kLong), 0.001, 0.001);
  EXPECT_EQ(equal, true);
};

TEST_F(EagerKernelTest, BroadCastIndexTest1) {
  torch::Tensor input_cpu = torch::arange(8).reshape({2, 2, 2});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{
      torch::tensor({1}),
      torch::tensor({0, 1}),
      torch::tensor({{0, 1}, {1, 1}})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  // auto tensorlist = indices.vec();
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }
  c10::List<c10::optional<at::Tensor>> indices_list{};
  // auto tensorlist = indices.vec();
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_list);

  // TODO: Check index_hpu_lazy why this long cast is required
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU).to(at::kLong));
  EXPECT_EQ(equal, true);
};

TEST_F(EagerKernelTest, Silu) {
  const std::vector<int64_t> dimentions{7, 3};
  auto input_tensor = torch::randn(dimentions, torch::requires_grad(false));
  auto hinput = input_tensor.to(torch::kHPU);

  auto hresult = torch::silu(hinput);
  auto hout = hresult.to(torch::kCPU);

  auto cpu_out = torch::silu(input_tensor);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(EagerKernelTest, Silu0Dim) {
  torch::Tensor A = torch::tensor(2.03);
  auto hinput = A.to(torch::kHPU);

  auto hresult = torch::silu(hinput);
  auto hout = hresult.to(torch::kCPU);

  auto cpu_out = torch::silu(A);
  EXPECT_TRUE(allclose(hout, cpu_out));
}

TEST_F(EagerKernelTest, RepeatTest) {
  torch::Tensor A = torch::randn({4, 5});

  torch::Tensor hA = A.to(torch::kHPU);
  torch::Tensor hOut = hA.repeat({2, 3});
  torch::Tensor Out = A.repeat({2, 3});

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}

TEST_F(EagerKernelTest, FlipTest) {
  torch::Tensor tensor = torch::rand({2, 3, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::flip(tHabana, {2, 1});
  auto out = torch::flip(tensor, {2, 1});
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, FlipNegativeTest) {
  torch::Tensor tensor = torch::randn({2, 2, 2});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::flip(tHabana, {-1, 1});
  auto out = torch::flip(tensor, {-1, 1});
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, Diag2DTest) {
  torch::Tensor tensor = torch::randn({4, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::diag(tHabana, 3);
  auto out = torch::diag(tensor, 3);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, Diag1DTest) {
  torch::Tensor tensor = torch::randn({3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);
  auto outHabana = torch::diag(tHabana, 1);
  auto out = torch::diag(tensor, 1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, DiagOut2DTest) {
  torch::Tensor tensor = torch::randn({4, 4});
  torch::Tensor tHabana = tensor.to(torch::kHPU);

  torch::Tensor out_tensor = torch::randn({1});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 3);
  auto out = torch::diag_out(out_tensor, tensor, 3);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, DiagOut1DTest) {
  torch::Tensor tensor = torch::randn({3});
  torch::Tensor tHabana = tensor.to(torch::kHPU);

  torch::Tensor out_tensor = torch::randn({4, 4});
  auto out_habana_tensor = out_tensor.to(torch::kHPU);

  auto outHabana = torch::diag_out(out_habana_tensor, tHabana, 1);
  auto out = torch::diag_out(out_tensor, tensor, 1);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST_F(EagerKernelTest, BatchNormBackwardAdd) {
  auto grad_tensor = torch::randn({10, 3, 4, 4}, torch::requires_grad(false));
  auto tHabanaGrad = grad_tensor.to(torch::kHPU);

  auto input_tensor = torch::randn({10, 3, 4, 4}, torch::requires_grad(false));
  auto tHabanaX = input_tensor.to(torch::kHPU);

  auto weight = torch::randn({3}, torch::requires_grad(false));
  auto tWeight = weight.to(torch::kHPU);

  auto mean = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaMean = mean.to(torch::kHPU);

  auto var = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaVar = var.to(torch::kHPU);

  auto save_mean = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaSaveMean = save_mean.to(torch::kHPU);

  auto save_ivar = torch::randn({3}, torch::requires_grad(false));
  auto tHabanaSaveIVar = save_ivar.to(torch::kHPU);

  auto results = torch::native_batch_norm_backward(
      tHabanaGrad,
      tHabanaX,
      tWeight,
      tHabanaMean,
      tHabanaVar,
      tHabanaSaveMean,
      tHabanaSaveIVar,
      true,
      0.1,
      {true, true, true});
  // to check caching
  results = torch::native_batch_norm_backward(
      tHabanaGrad,
      tHabanaX,
      tWeight,
      tHabanaMean,
      tHabanaVar,
      tHabanaSaveMean,
      tHabanaSaveIVar,
      true,
      0.1,
      {true, true, true});

  auto cpu_results = torch::native_batch_norm_backward(
      grad_tensor,
      input_tensor,
      weight,
      mean,
      var,
      save_mean,
      save_ivar,
      true,
      0.1,
      {true, true, true});
  tWeight = tWeight.add_(std::get<1>(results));
  weight = weight.add_(std::get<1>(cpu_results));

  EXPECT_EQ(allclose(tWeight.to(torch::kCPU), weight, 0.0001), true);
}

TEST_F(EagerKernelTest, LogSoftMaxTestBackward) {
  torch::Tensor input = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor grad = torch::rand({64, 10}, torch::requires_grad(false));
  torch::Tensor output = torch::rand({64, 10}, torch::requires_grad(false));

  torch::Tensor hinput = input.to(torch::kHPU);
  torch::Tensor hgrad = grad.to(torch::kHPU);
  torch::Tensor houtput = output.to(torch::kHPU);

  int dim = 0;
  auto hout_backward = torch::_log_softmax_backward_data(
      hgrad, houtput, dim, hinput.scalar_type());

  auto hout2_back = hout_backward.to(torch::kCPU);

  auto cout_back =
      _log_softmax_backward_data(grad, output, dim, input.scalar_type());

  EXPECT_EQ(allclose(hout2_back, cout_back), true);
}

TEST_F(EagerKernelTest, DISABLED_SumDimIntOut) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHPU);

  torch::Tensor hOut = at::empty_like(hA);
  torch::Tensor Out = at::empty_like(A);

  torch::Tensor out_cpu = torch::sum_outf(A, {0}, false, c10::nullopt, Out);
  torch::Tensor out_hpu = torch::sum_outf(hA, {0}, false, c10::nullopt, hOut);

  EXPECT_EQ(allclose(out_hpu.to(torch::kCPU), out_cpu), true);
}
