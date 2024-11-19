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

using namespace habana_lazy;
using namespace at;

class LazyIndexKernelTest : public habana_lazy_test::LazyTest {};
class UniqueParameterizedTestFixture
    : public ::testing::TestWithParam<
          std::tuple<torch::Tensor, c10::ScalarType, bool>>,
      public habana_lazy_test::EnvHelper {
  void SetUp() override {
    TearDownBridge();
  }
};
class UniqueDimParameterizedTestFixture
    : public ::testing::TestWithParam<
          std::tuple<torch::Tensor, c10::ScalarType, int64_t, bool, bool>>,
      public habana_lazy_test::EnvHelper {
  void SetUp() override {
    TearDownBridge();
  }
};

TEST_P(UniqueParameterizedTestFixture, DISABLED_tests) {
  // TODO: SW-172900
  c10::ScalarType dtype = std::get<1>(GetParam());
  torch::Tensor input_cpu = std::get<0>(GetParam()).to(dtype);
  bool return_inverse = std::get<2>(GetParam());

  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto out_hpu = torch::_unique(input_hpu, false, return_inverse);
  auto out_cpu = torch::_unique(input_cpu, true, return_inverse);

  auto unique_hpu_out = std::get<0>(out_hpu).to(torch::kCPU);
  auto unique_cpu_out = std::get<0>(out_cpu);
  // after upgrade to PT2.2 CPU supports only sorted mode
  // HPU supports only unsorted mode
  EXPECT_EQ(allclose(std::get<0>(unique_hpu_out.sort()), unique_cpu_out), true);
  // below code doesn`t work because of PT2.2 upgrade
  // if (return_inverse) {
  //   auto ri_hpu_out = std::get<1>(out_hpu).to(torch::kCPU);
  //   auto ri_cpu_out = std::get<1>(out_cpu);
  //   EXPECT_EQ(allclose(ri_hpu_out, ri_cpu_out), true);
  // }
}

INSTANTIATE_TEST_CASE_P(
    UniqueTest,
    UniqueParameterizedTestFixture,
    ::testing::Values(
        std::make_tuple(torch::randint(0, 10, {10}), torch::kInt32, false),
        std::make_tuple(torch::randint(0, 10, {200}), torch::kFloat, false),
        std::make_tuple(torch::randint(0, 10, {25, 25}), torch::kInt32, false),
        std::make_tuple(torch::randint(0, 10, {25, 25}), torch::kFloat, false),
        std::make_tuple(
            torch::randint(100, 200, {25, 20, 5}),
            torch::kInt32,
            false),
        std::make_tuple(torch::randint(20, 30, {10, 10}), torch::kFloat, false),
        std::make_tuple(
            torch::randint(50, 75, {5, 20, 20, 5}),
            torch::kInt32,
            false),
        std::make_tuple(
            torch::randint(50, 75, {5, 20, 20, 5}),
            torch::kFloat,
            false),
        std::make_tuple(torch::randint(0, 10, {10}), torch::kInt32, true),
        std::make_tuple(torch::randint(0, 10, {200}), torch::kFloat, true),
        std::make_tuple(torch::randint(0, 10, {25, 25}), torch::kInt32, true),
        std::make_tuple(torch::randint(0, 10, {25, 25}), torch::kFloat, true),
        std::make_tuple(
            torch::randint(100, 200, {25, 20, 5}),
            torch::kInt32,
            true),
        std::make_tuple(torch::randint(20, 30, {10, 10}), torch::kFloat, true),
        std::make_tuple(
            torch::randint(50, 75, {5, 20, 20, 5}),
            torch::kInt32,
            true),
        std::make_tuple(
            torch::randint(50, 75, {5, 20, 20, 5}),
            torch::kFloat,
            true)));

TEST_P(UniqueDimParameterizedTestFixture, tests) {
  c10::ScalarType dtype = std::get<1>(GetParam());
  int64_t dim = std::get<2>(GetParam());
  bool return_inverse = std::get<3>(GetParam());
  bool return_counts = std::get<4>(GetParam());

  torch::Tensor input_cpu = std::get<0>(GetParam()).to(dtype);
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  auto out_hpu =
      torch::unique_dim(input_hpu, dim, false, return_inverse, return_counts);
  auto out_cpu =
      torch::unique_dim(input_cpu, dim, false, return_inverse, return_counts);

  auto unique_hpu_out =
      std::get<0>(std::get<0>(out_hpu).to(torch::kCPU).sort(dim));
  auto unique_cpu_out = std::get<0>(std::get<0>(out_cpu).sort(dim));

  EXPECT_EQ(allclose(unique_hpu_out, unique_cpu_out), true);

  if (return_inverse) {
    auto ri_hpu_out = std::get<0>(std::get<1>(out_hpu).to(torch::kCPU).sort(0));
    auto ri_cpu_out = std::get<0>(std::get<1>(out_cpu).sort(0));
    EXPECT_EQ(allclose(ri_hpu_out, ri_cpu_out), true);
  }

  if (return_counts) {
    auto rc_hpu_out = std::get<0>(std::get<2>(out_hpu).to(torch::kCPU).sort(0));
    auto rc_cpu_out = std::get<0>(std::get<2>(out_cpu).sort(0));
    EXPECT_EQ(allclose(rc_hpu_out, rc_cpu_out), true);
  }
}

INSTANTIATE_TEST_CASE_P(
    UniqueDimTest,
    UniqueDimParameterizedTestFixture,
    ::testing::Values(
        std::make_tuple( // 0
            torch::tensor(
                {{{1, 2, 3, 2, 1}, {4, 5, 6, 5, 4}, {1, 2, 3, 2, 1}},
                 {{1, 2, 3, 2, 1}, {4, 5, 6, 5, 4}, {1, 2, 3, 2, 1}}}),
            torch::kInt32,
            -1,
            false,
            false),
        std::make_tuple( // 1
            torch::tensor({{1, 2, 3, 2, 1}, {4, 5, 6, 5, 4}, {1, 2, 3, 2, 1}}),
            torch::kInt32,
            1,
            true,
            true),
        std::make_tuple( // 2
            torch::tensor({{1, 2, 3, 2, 1}, {4, 5, 6, 7, 4}, {1, 2, 3, 2, 1}}),
            torch::kInt32,
            1,
            true,
            true),
        std::make_tuple( // 3
            torch::tensor({{1, 2, 3, 2, 1}, {4, 5, 6, 5, 4}, {1, 2, 3, 2, 1}}),
            torch::kInt32,
            1,
            false,
            false),
        std::make_tuple( // 4
            torch::tensor({{1, 2, 3, 2, 1}, {4, 5, 6, 7, 4}, {1, 2, 3, 2, 1}}),
            torch::kInt32,
            1,
            false,
            true),
        std::make_tuple( // 5
            torch::tensor({{1, 2, 3, 2, 1}, {4, 5, 6, 7, 4}, {1, 2, 3, 2, 1}}),
            torch::kInt32,
            1,
            true,
            false),
        std::make_tuple( // 6
            torch::tensor(
                {{{1, 2, 3, 2, 1}, {4, 5, 6, 5, 4}, {1, 2, 3, 2, 1}},
                 {{1, 2, 3, 2, 1}, {4, 5, 6, 5, 4}, {1, 2, 3, 2, 1}}}),
            torch::kInt32,
            2,
            false,
            false),
        std::make_tuple( // 7
            torch::randint(0, 10, {10}),
            torch::kInt32,
            0,
            false,
            true),
        std::make_tuple( // 8
            torch::randint(0, 10, {20}),
            torch::kFloat,
            0,
            false,
            true),
        std::make_tuple( // 9
            torch::randint(0, 10, {10, 20}),
            torch::kInt32,
            0,
            true,
            false),
        std::make_tuple( // 10
            torch::randint(0, 10, {10, 20}),
            torch::kFloat,
            0,
            false,
            false),
        std::make_tuple( // 11
            torch::randint(0, 10, {10, 20}),
            torch::kInt32,
            1,
            false,
            false),
        std::make_tuple( // 12
            torch::randint(0, 10, {10, 20}),
            torch::kFloat,
            1,
            true,
            true),
        std::make_tuple( // 13
            torch::randint(0, 10, {10, 20, 10}),
            torch::kInt32,
            0,
            false,
            true),
        std::make_tuple( // 14
            torch::randint(0, 10, {15, 12, 13}),
            torch::kFloat,
            0,
            true,
            false),
        std::make_tuple( // 15
            torch::randint(0, 10, {13, 24, 21}),
            torch::kInt32,
            1,
            false,
            false),
        std::make_tuple( // 16
            torch::randint(0, 10, {23, 14, 25}),
            torch::kFloat,
            1,
            true,
            true),
        std::make_tuple( // 17
            torch::randint(0, 10, {13, 24, 3}),
            torch::kInt32,
            2,
            false,
            true),
        std::make_tuple( // 18
            torch::randint(0, 10, {4, 14, 11}),
            torch::kFloat,
            2,
            false,
            false),

        std::make_tuple( // 19
            torch::randint(0, 10, {10, 2, 10, 1}),
            torch::kInt32,
            0,
            false,
            false),
        std::make_tuple( // 20
            torch::randint(0, 10, {7, 12, 13, 7}),
            torch::kFloat,
            0,
            true,
            true),
        std::make_tuple( // 21
            torch::randint(0, 10, {13, 1, 21, 12}),
            torch::kInt32,
            1,
            false,
            false),
        std::make_tuple( // 22
            torch::randint(0, 10, {23, 11, 5, 7}),
            torch::kFloat,
            1,
            false,
            true),
        std::make_tuple( // 23
            torch::randint(0, 10, {13, 24, 7, 13}),
            torch::kInt32,
            2,
            false,
            false),
        std::make_tuple( // 24
            torch::randint(0, 10, {23, 14, 2, 4}),
            torch::kFloat,
            2,
            true,
            false),
        std::make_tuple( // 25
            torch::randint(0, 10, {3, 2, 7, 22}),
            torch::kInt32,
            3,
            true,
            true),
        std::make_tuple( // 26
            torch::randint(0, 10, {1, 13, 2, 21}),
            torch::kFloat,
            3,
            false,
            true)));

TEST_F(LazyIndexKernelTest, IndexAddInplaceTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  auto index = torch::tensor({0, 1}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);
  auto source = torch::randn({8, 2, 28, 28}, torch::requires_grad(false));
  auto h_source = source.to(torch::kHPU);

  h_a.index_add_(dim, h_index, h_source);
  auto h_temp = torch::zeros({8, 3, 28, 28}).to(torch::kHPU);
  auto out = torch::add(h_a, h_temp);

  auto h_cout = out.to(torch::kCPU);

  a.index_add_(dim, index, source);

  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, IndexAddOutTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  auto index = torch::tensor({0, 1}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);
  auto source = torch::randn({8, 2, 28, 28}, torch::requires_grad(false));
  auto h_source = source.to(torch::kHPU);

  auto h_out = torch::tensor({}, torch::requires_grad(false)).to(torch::kHPU);
  h_out = at::index_add_out(h_out, h_a, dim, h_index, h_source);

  auto h_temp = torch::zeros({8, 3, 28, 28}).to(torch::kHPU);
  auto out = torch::add(h_out, h_temp);

  auto h_cout = out.to(torch::kCPU);

  a.index_add_(dim, index, source);

  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, IndexCopyInplaceTest_2) {
  auto index_copy = [](int64_t dim) {
    torch::Tensor x = torch::zeros({5, 5});
    torch::Tensor h_x = x.to(torch::kHPU);
    torch::Tensor source = torch::tensor(
        {{1, 2, 3, 4, 5},
         {6, 7, 8, 9, 10},
         {11, 12, 13, 14, 15},
         {16, 17, 18, 19, 20},
         {21, 22, 23, 24, 25}},
        torch::dtype(torch::kFloat));
    torch::Tensor h_source = source.to(torch::kHPU);
    torch::Tensor index = torch::tensor({0, 4, 2, 3, 1});
    torch::Tensor h_index = index.to(torch::kHPU);
    x.index_copy_(dim, index, source);
    h_x.index_copy_(dim, h_index, h_source);
    EXPECT_EQ(allclose(x, h_x.to(torch::kCPU)), true);
  };
  index_copy(1);
  index_copy(0);
}

TEST_F(LazyIndexKernelTest, ScatterValueInplaceTest) {
  torch::Tensor a = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 0;
  auto index = torch::randint(0, 5, {5, 7}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);
  auto value = 2;

  h_a.scatter_(dim, h_index, value);
  auto h_cout = h_a.to(torch::kCPU);
  a.scatter_(dim, index, value);

  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, ScatterTest) {
  torch::Tensor a = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 0;
  auto index = torch::randint(0, 5, {5, 7}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);
  torch::Tensor src = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_src = src.to(torch::kHPU);

  torch::Tensor hOut = torch::scatter(h_a, dim, h_index, h_src);
  auto h_cout = hOut.to(torch::kCPU);
  torch::Tensor out = torch::scatter(a, dim, index, src);

  EXPECT_EQ(allclose(h_cout, out), true);
}

TEST_F(LazyIndexKernelTest, ScatterMoveTest) {
  torch::Tensor a =
      torch::tensor({{0, 1}, {1, 0}}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  torch::Tensor r = torch::tensor(
      {{false, true}, {false, true}}, torch::requires_grad(false));
  torch::Tensor r_a = r.to(torch::kHPU);

  torch::Tensor hOut = torch::scatter(r_a, 1, h_a, r_a);
  auto h_cout = hOut.to(torch::kCPU);
  torch::Tensor out = torch::scatter(r, 1, a, r);

  EXPECT_EQ(allclose(h_cout, out), true);
}

TEST_F(LazyIndexKernelTest, ArangeFloatOutTest) {
  torch::Tensor tStart = torch::tensor(0.0);
  torch::Tensor tEnd = torch::tensor(10.0);
  torch::Tensor tStep = torch::tensor(0.25);
  torch::Scalar start = tStart.item();
  torch::Scalar end = tEnd.item();
  torch::Scalar step = tStep.item();

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Float;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);

  auto h_a = torch::arange(start, end, step, hb_options);
  auto h_cout = h_a.to(torch::kCPU);
  auto a = torch::arange(start, end, step, cpu_options);
  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, ArangeIntOutTest) {
  torch::Tensor tStart = torch::tensor(0);
  torch::Tensor tEnd = torch::tensor(10);
  torch::Tensor tStep = torch::tensor(1);
  torch::Scalar start = tStart.item();
  torch::Scalar end = tEnd.item();
  torch::Scalar step = tStep.item();

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);

  auto h_a = torch::arange(start, end, step, hb_options);
  auto h_cout = h_a.to(torch::kCPU);
  auto a = torch::arange(start, end, step, cpu_options);
  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, ArangeCharOutTest) {
  torch::Tensor tStart = torch::tensor(0);
  torch::Tensor tEnd = torch::tensor(10);
  torch::Tensor tStep = torch::tensor(1);
  torch::Scalar start = tStart.item();
  torch::Scalar end = tEnd.item();
  torch::Scalar step = tStep.item();

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Char;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);

  auto h_a = torch::arange(start, end, step, hb_options);
  auto h_cout = h_a.to(torch::kCPU);
  auto a = torch::arange(start, end, step, cpu_options);
  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, IndexTest) {
  torch::Tensor input_cpu = torch::arange(9).reshape({3, 3});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{
      torch::tensor({0, 1}), torch::tensor({0, 1})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  // auto tensorlist = indices.vec();
  indices_cpu.reserve(vec_cpu.size());
  for (const auto& t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }

  // auto out_cpu = at::index(input_cpu, vec_cpu).to(torch::kInt32);
  // auto out_hpu = at::index(input_hpu, vec_hpu);
  c10::List<c10::optional<at::Tensor>> indices_list{};
  // auto tensorlist = indices.vec();
  indices_list.reserve(vec_cpu.size());
  for (const auto& t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto A = torch::randint(0, 5, {2}, torch::dtype(torch::kInt64));
  auto add_out = at::add(out_cpu, A);

  auto out_hpu = at::index(input_hpu, indices_list);
  auto add_out_hpu = at::add(out_hpu, A.to(torch::kHPU));
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
  bool equal_add = add_out.allclose(add_out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal_add, true);
}

TEST_F(LazyIndexKernelTest, ArangeLongOutTest) {
  torch::Tensor tStart = torch::tensor(0);
  torch::Tensor tEnd = torch::tensor(10);
  torch::Tensor tStep = torch::tensor(1);
  torch::Scalar start = tStart.item();
  torch::Scalar end = tEnd.item();
  torch::Scalar step = tStep.item();

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Long;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);
  auto h_a = torch::arange(start, end, step, hb_options);
  auto h_cout = h_a.to(torch::kCPU);
  auto a = torch::arange(start, end, step, cpu_options);
  EXPECT_EQ(allclose(h_cout, a), true);
}

TEST_F(LazyIndexKernelTest, NonZeroTestMixValues) {
  torch::Tensor input_cpu =
      torch::randint(0, 7, {5, 7}, torch::dtype(torch::kInt64));
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto out_hpu = torch::nonzero(input_hpu);
  auto out_cpu = torch::nonzero(input_cpu);
  auto h_cout = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(h_cout, out_cpu), true);
}

TEST_F(LazyIndexKernelTest, NonZeroTestMixValues_CmptOpShp) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);

  torch::Tensor input_cpu =
      torch::randint(0, 7, {5, 7}, torch::dtype(torch::kInt64));
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto out_hpu = torch::nonzero(input_hpu);
  auto out_cpu = torch::nonzero(input_cpu);
  auto h_cout = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(h_cout, out_cpu), true);

  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyIndexKernelTest, NonZeroV2TestMixValues) {
  if (false == GET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE))
    SET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE, true, 1);
  auto test = [](std::vector<int64_t> shape, c10::ScalarType type) {
    torch::Tensor input_cpu = torch::randint(0, 2, shape, torch::dtype(type));
    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
    auto out_hpu = torch::nonzero(input_hpu);
    auto out_cpu = torch::nonzero(input_cpu);
    auto h_cout = out_hpu.to(torch::kCPU);
    EXPECT_EQ(allclose(h_cout, out_cpu), true);
  };
  test({2, 121}, torch::kInt8);
  test({2, 121}, torch::kInt32);
  test({2, 121}, torch::kFloat);
  test({2, 121}, torch::kBFloat16);
  test({2, 3, 121}, torch::kInt8);
  UNSET_ENV_FLAG_NEW(PT_HPU_VALIDATE_COMPUTE_SHAPE);
}

TEST_F(LazyIndexKernelTest, NonZeroTestAllFalse) {
  torch::Tensor input_cpu =
      torch::randint(0, 1, {5, 7}, torch::dtype(torch::kInt64));
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto out_hpu = torch::nonzero(input_hpu);
  auto out_cpu = torch::nonzero(input_cpu);
  auto h_cout = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(h_cout, out_cpu), true);
}

TEST_F(LazyIndexKernelTest, NonZeroTestAllFalse0D) {
  torch::Tensor input_cpu = torch::tensor({});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto out_hpu = torch::nonzero(input_hpu);
  auto out_cpu = torch::nonzero(input_cpu);
  auto h_cout = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(h_cout, out_cpu), true);
}

TEST_F(LazyIndexKernelTest, NonZeroOutTestMixValues) {
  torch::Tensor input_cpu =
      torch::randint(0, 7, {5, 7}, torch::dtype(torch::kInt64));
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  torch::Tensor hOut = at::empty_like(input_hpu);
  torch::Tensor out_cpu = at::empty_like(input_cpu);
  torch::nonzero_outf(input_cpu, out_cpu);
  torch::nonzero_outf(input_hpu, hOut);
  auto out_hpu = hOut.to(torch::kCPU);
  EXPECT_EQ(allclose(out_hpu, out_cpu), true);
}

TEST_F(LazyIndexKernelTest, UniqueTest) {
  auto typetest = [](c10::ScalarType dtype) {
    torch::Tensor input_cpu = torch::randint(0, 10, {1, 2, 2, 3}).to(dtype);
    torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
    auto out_hpu = std::get<0>(torch::_unique2(input_hpu, false, false, false));
    auto out_cpu = std::get<0>(torch::_unique2(input_cpu, false, false, false));
    auto h_cout = out_hpu.to(torch::kCPU);
    EXPECT_EQ(
        allclose(
            std::get<0>(h_cout.view(-1).sort()),
            std::get<0>(out_cpu.view(-1).sort())),
        true);
  };
  typetest(torch::kInt32);
  typetest(torch::kLong);
}

TEST_F(LazyIndexKernelTest, LinspaceTestStep1) {
  torch::Scalar start = 0.0;
  torch::Scalar end = 10.0;
  long int step = 11;

  auto h_a = torch::linspace(start, end, step);
  auto hOut = h_a.to(torch::kCPU);

  auto a = torch::linspace(start, end, step);
  EXPECT_EQ(allclose(hOut, a), true);
}

TEST_F(LazyIndexKernelTest, LinspaceTestDivisableByStep) {
  torch::Scalar start = 612.3;
  torch::Scalar end = 630.3;
  long int step = 7;

  auto h_a = torch::linspace(start, end, step);
  auto hOut = h_a.to(torch::kCPU);

  auto a = torch::linspace(start, end, step);
  EXPECT_EQ(allclose(hOut, a), true);
}

TEST_F(LazyIndexKernelTest, LinspaceTestDivisableByStepFractionalRange) {
  torch::Scalar start = 0.00093;
  torch::Scalar end = 0.00373;
  long int step = 8;

  auto h_a = torch::linspace(start, end, step);
  auto hOut = h_a.to(torch::kCPU);

  auto a = torch::linspace(start, end, step);
  EXPECT_EQ(allclose(hOut, a), true);
}

TEST_F(LazyIndexKernelTest, LinspaceOutPosToNeFraction) {
  const int64_t constStepsValue = 45;
  torch::Scalar start = 0.70f;
  torch::Scalar end = -0.03f;
  int64_t step = constStepsValue;
  torch::Tensor out =
      torch::randn({constStepsValue}, torch::requires_grad(false));
  auto hOut = out.to(torch::kHPU);

  auto h_a = torch::linspace_outf(start, end, step, hOut);
  auto hOut_cpu = h_a.to(torch::kCPU);

  auto a = torch::linspace_outf(start, end, step, out);
  EXPECT_EQ(allclose(hOut_cpu, out, 0.0001), true);
}

TEST_F(LazyIndexKernelTest, LinspaceOutSameStartEnd) {
  torch::Scalar start = -100.0f;
  torch::Scalar end = -100.0f;
  int64_t step = 100; // wrong value
  torch::Tensor out = torch::randn({10}, torch::requires_grad(false));
  auto hOut = out.to(torch::kHPU);

  auto h_a = torch::linspace_outf(start, end, step, hOut);
  auto hOut_cpu = h_a.to(torch::kCPU);

  auto a = torch::linspace_outf(start, end, step, out);
  EXPECT_EQ(allclose(hOut_cpu, out), true);
}

TEST_F(LazyIndexKernelTest, SelectNDimsTest) {
  // Slice H2D flow only supports max 5dims : SW-153474
  SET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE, false, 1);
  torch::Tensor a =
      torch::randn({2, 3, 4, 5, 6, 4}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 2;

  Tensor h_out = torch::select(h_a, dim, 3);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select(a, dim, 3);

  EXPECT_EQ(allclose(h_cout, cout), true);
  UNSET_ENV_FLAG_NEW(PT_HPU_ENABLE_H2D_DYNAMIC_SLICE);
}

TEST_F(LazyIndexKernelTest, squeezeTest) {
  auto x = torch::randn({4});
  auto hx = x.to(torch::kHPU);

  auto B = torch::squeeze(x);
  auto hB = torch::squeeze(hx);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyIndexKernelTest, squeezeTestNodim) {
  auto x = torch::randn({1, 2, 1, 3});
  auto hx = x.to(torch::kHPU);

  auto B = torch::squeeze(x);
  auto hB = torch::squeeze(hx);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyIndexKernelTest, squeezeDimsAll) {
  auto x = torch::randn({3, 1, 7, 4, 1});
  std::vector<int64_t> dims{1, 4};
  auto hx = x.to(torch::kHPU);

  auto B = torch::squeeze(x, dims);
  auto hB = torch::squeeze(hx, dims);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyIndexKernelTest, squeezeDimsOne) {
  auto x = torch::randn({1, 1, 7, 4, 1});
  std::vector<int64_t> dims{1};
  auto hx = x.to(torch::kHPU);

  auto B = torch::squeeze(x, dims);
  auto hB = torch::squeeze(hx, dims);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyIndexKernelTest, squeezeDimsNone) {
  auto x = torch::randn({1, 1, 7, 4, 1});
  std::vector<int64_t> dims{2, 3};
  auto hx = x.to(torch::kHPU);

  auto B = torch::squeeze(x, dims);
  auto hB = torch::squeeze(hx, dims);

  EXPECT_EQ(allclose(B, hB.cpu(), 0.001, 0.001), true);
}

TEST_F(LazyIndexKernelTest, IndexOutTest) {
  // SET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE, 0, 0);
  torch::Tensor input_cpu = torch::arange(36).reshape({4, 3, 3});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{torch::tensor({0, 1}), torch::tensor({1})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.reserve(vec_cpu.size() + 1);
  at::Tensor undef_t;
  indices_cpu.push_back(undef_t);
  for (const auto& t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }

  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.reserve(vec_cpu.size() + 1);
  indices_list.push_back(undef_t);
  for (const auto& t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  std::vector<int64_t> out_size = {4, 2};
  torch::ScalarType dtype = input_cpu.scalar_type();
  auto expected = torch::empty(out_size, dtype);
  auto res = torch::empty(out_size, torch::TensorOptions(dtype).device("hpu"));

  auto out_cpu = at::index_out(expected, input_cpu, indices_cpu);

  auto out_hpu = at::index_out(res, input_hpu, indices_list);
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyIndexKernelTest, IndexMixedTest1) {
  torch::Tensor input_cpu = torch::arange(36).reshape({4, 3, 3});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu;
  torch::Tensor index_t_cpu = torch::tensor({0, 1, 2, 0, 1, 2});
  torch::Tensor bmask_cpu = torch::tensor(
      {{false, true, false},
       {false, false, true},
       {true, true, false},
       {false, true, true}});
  /*Index as input[bmask_cpu, index_t_cpu]*/
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.emplace_back(bmask_cpu);
  indices_cpu.emplace_back(index_t_cpu);

  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.push_back(c10::make_optional(bmask_cpu.to(torch::kHPU)));
  indices_list.push_back(c10::make_optional(index_t_cpu.to(torch::kHPU)));
  std::vector<int64_t> out_size = {6};
  torch::ScalarType dtype = input_cpu.scalar_type();
  auto expected = torch::empty(out_size, dtype);
  auto res = torch::empty(out_size, torch::TensorOptions(dtype).device("hpu"));

  auto out_cpu = at::index_out(expected, input_cpu, indices_cpu);

  auto out_hpu = at::index_out(res, input_hpu, indices_list);
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyIndexKernelTest, IndexMixedTest2) {
  torch::Tensor input_cpu = torch::arange(108).reshape({4, 3, 3, 3});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu;
  torch::Tensor index_t_cpu = torch::tensor({0, 1, 2, 0, 1, 2});
  torch::Tensor bmask_cpu = torch::tensor(
      {{false, true, false},
       {false, false, true},
       {true, true, false},
       {false, true, true}});
  /*Index as input[bmask_cpu, :, index_t_cpu]*/
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.emplace_back(bmask_cpu);
  at::Tensor undef_t;
  indices_cpu.push_back(undef_t);
  indices_cpu.emplace_back(index_t_cpu);

  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.push_back(c10::make_optional(bmask_cpu.to(torch::kHPU)));
  indices_list.push_back(undef_t);
  indices_list.push_back(c10::make_optional(index_t_cpu.to(torch::kHPU)));
  std::vector<int64_t> out_size = {6, 3};
  torch::ScalarType dtype = input_cpu.scalar_type();
  auto expected = torch::empty(out_size, dtype);
  auto res = torch::empty(out_size, torch::TensorOptions(dtype).device("hpu"));

  auto out_cpu = at::index_out(expected, input_cpu, indices_cpu);

  auto out_hpu = at::index_out(res, input_hpu, indices_list);
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyIndexKernelTest, IndexMultiDimTest) {
  torch::Tensor input_cpu = torch::arange(36).reshape({4, 3, 3});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu;
  torch::Tensor index_t_cpu = torch::tensor({{0}});
  /*Index as input[bmask_cpu, index_t_cpu]*/
  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.emplace_back(index_t_cpu);

  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.push_back(c10::make_optional(index_t_cpu.to(torch::kHPU)));
  auto out_cpu = at::index(input_cpu, indices_cpu);

  auto out_hpu = at::index(input_hpu, indices_list);
  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyIndexKernelTest, IndexPutNegativeIndicesTest) {
  torch::Tensor input_cpu = torch::arange(16).reshape({4, 4});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  torch::Tensor value_cpu = torch::tensor(-100);
  torch::Tensor value_hpu = value_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{
      torch::tensor({0, 1, 2}), torch::tensor({0, -2, -1})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  indices_cpu.reserve(vec_cpu.size());
  for (const auto& t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }

  c10::List<c10::optional<at::Tensor>> indices_list{};
  indices_list.reserve(vec_cpu.size());
  for (const auto& t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu1 =
      at::_index_put_impl_(input_cpu, indices_cpu, value_cpu, true, false);
  setenv("PT_HPU_ENABLE_NEGATIVE_INDEXING", "true", 1);
  auto out_hpu1 =
      at::_index_put_impl_(input_hpu, indices_list, value_hpu, true, false);
  bool equal1 = out_cpu1.allclose(out_hpu1.to(torch::kCPU), 0.001, 0.001);
  unsetenv("PT_HPU_ENABLE_NEGATIVE_INDEXING");
  EXPECT_EQ(equal1, true);

  auto out_cpu2 =
      at::_index_put_impl_(input_cpu, indices_cpu, value_cpu, false, false);
  auto out_hpu2 =
      at::_index_put_impl_(input_hpu, indices_list, value_hpu, false, false);
  bool equal2 = out_cpu2.allclose(out_hpu2.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal2, true);
}
