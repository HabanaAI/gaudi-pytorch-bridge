#include <gtest/gtest.h>
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
#include "habana_lazy_test_infra.h"

using namespace habana_lazy;

class LazyTensorShapeKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyTensorShapeKernelTest, CatExecTest1) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  auto C = torch::relu(A);
  auto D = torch::relu(B);
  auto exp = torch::cat({C, D});

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);

  auto hC = torch::relu(hA);
  auto hD = torch::relu(hB);

  torch::Tensor out = torch::cat({hC, hD});
  auto result = out.to(torch::kCPU);
  EXPECT_EQ(allclose(result, exp), true);
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest2) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  auto exp = torch::cat({A, B});

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);

  torch::Tensor out = torch::cat({hA, hB});
  auto result = out.to(torch::kCPU);
  EXPECT_EQ(allclose(result, exp), true);
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest3) {
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 4}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));

  auto tempc1 = torch::cat({A, B}, 1);
  auto exp = torch::cat({A, tempc1}, 1);

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = B.to(torch::kHABANA);

  torch::Tensor temp1 = torch::cat({hA, hB}, 1);
  auto out = torch::cat({hA, temp1}, 1);

  auto result = out.to(torch::kCPU);

  EXPECT_EQ(allclose(result, exp), true);
}

TEST_F(LazyTensorShapeKernelTest, PermuteTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = hA.permute({1, 0});
  torch::Tensor Out = A.permute({1, 0});

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, TTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::t(hA);
  torch::Tensor Out = torch::t(A);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, SelectTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);

  int64_t dim = 1;
  int64_t index = 0;

  Tensor h_out = torch::select(h_a, dim, index);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select(a, dim, index);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, SliceTest) {
  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);
  int64_t dim = 1;
  int64_t start_index = 0;
  int64_t end = 8;
  int64_t step = 1;

  Tensor h_out = torch::slice(h_a, dim, start_index, end, step);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::slice(a, dim, start_index, end, step);

  EXPECT_EQ(allclose(h_cout, cout), true);
}

TEST_F(LazyTensorShapeKernelTest, ViewExecute) {
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanain = input_tensor.to(torch::kHABANA);
  c10::IntArrayRef new_size = {-1, 48};
  auto result = torch::_unsafe_view(tHabanain, new_size);
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(result));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors);
  at::Tensor result_lazy = result.to(torch::kCPU);
  auto result_cpu = torch::_unsafe_view(input_tensor, new_size);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest) {
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::transpose(hA, 1, 0);
  torch::Tensor Out = torch::transpose(A, 1, 0);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, ExpandTest) {
  torch::Tensor A = torch::randn({3, 1}, torch::requires_grad(false));

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = hA.expand({3, 4}, false);
  torch::Tensor Out = A.expand({3, 4}, false);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
}

TEST_F(LazyTensorShapeKernelTest, Repeat) {
  torch::Tensor A = torch::randn({4, 5});

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = hA.repeat({2, 3});
  torch::Tensor Out = A.repeat({2, 3});

  EXPECT_TRUE(allclose(hOut.to(torch::kCPU), Out));
}
