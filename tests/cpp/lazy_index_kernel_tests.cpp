#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
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
using namespace at;

class LazyIndexKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyIndexKernelTest, IndexSelectTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  auto index = torch::tensor({0, 1}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);

  Tensor h_out = torch::index_select(h_a, dim, h_index);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::index_select(a, dim, index);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

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

TEST_F(LazyIndexKernelTest, Onehot) {
  auto onehot = [](std::string device, int64_t num_classes) {
    auto t = (torch::arange(20) % 4).view({4, 5}).to(device);
    auto result = torch::one_hot(t, num_classes);
    return result.to("cpu");
  };

  EXPECT_TRUE(allclose(onehot("cpu", 4), onehot("hpu", 4)));
  EXPECT_TRUE(allclose(onehot("cpu", -1), onehot("hpu", -1)));
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

TEST_F(LazyIndexKernelTest, ScatterValueTest) {
  torch::Tensor a = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 0;
  auto index = torch::randint(0, 5, {5, 7}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);
  auto value = 2;

  torch::Tensor hOut = torch::scatter(h_a, dim, h_index, value);
  auto h_cout = hOut.to(torch::kCPU);
  torch::Tensor out = torch::scatter(a, dim, index, value);

  EXPECT_EQ(allclose(h_cout, out), true);
}

// This test is failing randomly.
// https://jira.habana-labs.com/browse/SW-44742

/*TEST_F(LazyIndexKernelTest, ScatterAddTest) {
  torch::Tensor a = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHPU);
  int64_t dim = 1;
  auto index = torch::randint(0, 5, {5, 7}, torch::dtype(torch::kInt64));
  auto h_index = index.to(torch::kHPU);
  torch::Tensor src = torch::randn({5, 7}, torch::requires_grad(false));
  torch::Tensor h_src = src.to(torch::kHPU);

  torch::Tensor hOut = torch::scatter_add(h_a, dim, h_index, h_src);
  auto h_cout = hOut.to(torch::kCPU);
  torch::Tensor out = torch::scatter_add(a, dim, index, src);

  EXPECT_EQ(allclose(h_cout, out), true);
}*/

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

TEST_F(LazyIndexKernelTest, IndexTest) {
  torch::Tensor input_cpu = torch::arange(9).reshape({3, 3});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  std::vector<torch::Tensor> vec_cpu{
      torch::tensor({0, 1}), torch::tensor({0, 1})};

  c10::List<c10::optional<at::Tensor>> indices_cpu{};
  // auto tensorlist = indices.vec();
  indices_cpu.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_cpu.push_back(c10::make_optional(t));
  }

  // auto out_cpu = at::index(input_cpu, vec_cpu).to(torch::kInt32);
  // auto out_hpu = at::index(input_hpu, vec_hpu);
  c10::List<c10::optional<at::Tensor>> indices_list{};
  // auto tensorlist = indices.vec();
  indices_list.reserve(vec_cpu.size());
  for (auto t : vec_cpu) {
    indices_list.push_back(c10::make_optional(t.to(torch::kHPU)));
  }
  auto out_cpu = at::index(input_cpu, indices_cpu).to(torch::kInt32);
  auto out_hpu = at::index(input_hpu, indices_list);

  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
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
  auto out_cpu = torch::nonzero(input_cpu).to(torch::kInt32);
  auto h_cout = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(h_cout, out_cpu), true);
}

TEST_F(LazyIndexKernelTest, NonZeroTestAllFalse) {
  torch::Tensor input_cpu =
      torch::randint(0, 1, {5, 7}, torch::dtype(torch::kInt64));
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);
  auto out_hpu = torch::nonzero(input_hpu);
  auto out_cpu = torch::nonzero(input_cpu).to(torch::kInt32);
  auto h_cout = out_hpu.to(torch::kCPU);
  EXPECT_EQ(allclose(h_cout, out_cpu), true);
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

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);

  auto h_a = torch::linspace(start, end, step);
  auto hOut = h_a.to(torch::kCPU);

  auto a = torch::linspace(start, end, step);
  EXPECT_EQ(allclose(hOut, a), true);
}

TEST_F(LazyIndexKernelTest, LinspaceTestDivisableByStep) {
  torch::Scalar start = 612.3;
  torch::Scalar end = 630.3;

  long int step = 7;

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);

  auto h_a = torch::linspace(start, end, step);
  auto hOut = h_a.to(torch::kCPU);

  auto a = torch::linspace(start, end, step);
  EXPECT_EQ(allclose(hOut, a), true);
}

TEST_F(LazyIndexKernelTest, LinspaceTestDivisableByStepFractionalRange) {
  torch::Scalar start = 0.00093;
  torch::Scalar end = 0.00373;

  long int step = 8;

  c10::optional<at::ScalarType> dtype = c10::ScalarType::Int;

  c10::optional<at::Device> hb_device = at::DeviceType::HPU;
  at::TensorOptions hb_options =
      at::TensorOptions().dtype(dtype).device(hb_device);
  c10::optional<at::Device> cpu_device = at::DeviceType::CPU;
  at::TensorOptions cpu_options =
      at::TensorOptions().dtype(dtype).device(cpu_device);

  auto h_a = torch::linspace(start, end, step);
  auto hOut = h_a.to(torch::kCPU);

  auto a = torch::linspace(start, end, step);
  EXPECT_EQ(allclose(hOut, a), true);
}

TEST_F(LazyIndexKernelTest, AdvanceIndexTest) {
  torch::Tensor input_cpu = torch::arange(48).reshape({8, 6});
  torch::Tensor input_hpu = input_cpu.to(torch::kHPU);

  auto i1 = torch::Tensor();
  auto i2 = torch::tensor({4, 5});
  c10::List<c10::optional<at::Tensor>> indices_cpu{
      c10::make_optional(i1), c10::make_optional(i2)};

  c10::List<c10::optional<at::Tensor>> indices_hpu{
      c10::make_optional(i1), c10::make_optional(i2.to(torch::kHPU))};

  auto out_cpu = at::index(input_cpu, indices_cpu);
  auto out_hpu = at::index(input_hpu, indices_hpu);

  bool equal = out_cpu.allclose(out_hpu.to(torch::kCPU), 0.001, 0.001);
  EXPECT_EQ(equal, true);
}

TEST_F(LazyIndexKernelTest, LinspaceOutPosToNeFraction) {
  const int64_t constStepsValue = 45;
  torch::Scalar start = 0.70f;
  torch::Scalar end = -0.03f;
  c10::optional<int64_t> step = constStepsValue;
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
  c10::optional<int64_t> step = 100; // wrong value
  torch::Tensor out = torch::randn({10}, torch::requires_grad(false));
  auto hOut = out.to(torch::kHPU);

  auto h_a = torch::linspace_outf(start, end, step, hOut);
  auto hOut_cpu = h_a.to(torch::kCPU);

  auto a = torch::linspace_outf(start, end, step, out);
  EXPECT_EQ(allclose(hOut_cpu, out), true);
}
