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

using namespace habana_lazy;

class LazyTensorShapeKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(LazyTensorShapeKernelTest, CatTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor C = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);
  torch::Tensor hC = C.to(torch::kHABANA);
  torch::Tensor out = torch::cat({hA, hB, hC});

  auto hl_result = GetHbLazyTensor(out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check("Tensor[] = prim::ListConstruct")
      ->check("int = prim::Constant[value=0]")
      ->check("Tensor = aten::cat")
      ->run(*exec.get_graph());
  // ASSERT_TRUE(torch::allclose(hz_exp, hz_exp));
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest1) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
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
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, CatExecTest2) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 2}, torch::requires_grad(false));
  torch::Tensor B = torch::randn({2, 2}, torch::requires_grad(false));

  auto exp = torch::cat({A, B});

  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hB = B.to(torch::kHABANA);

  torch::Tensor out = torch::cat({hA, hB});
  auto result = out.to(torch::kCPU);
  EXPECT_EQ(allclose(result, exp), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, PermuteTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = hA.permute({1, 0});
  torch::Tensor Out = A.permute({1, 0});

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors, {});

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, TTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::t(hA);
  torch::Tensor Out = torch::t(A);

  std::vector<HbLazyTensor> hl_tensors = {GetHbLazyTensor(hOut)};
  HbLazyTensor::SyncTensorsGraph(&hl_tensors, {});

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, SelectTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);

  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);

  int64_t dim = 1;
  int64_t index = 0;

  Tensor h_out = torch::select(h_a, dim, index);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::select(a, dim, index);

  EXPECT_EQ(allclose(h_cout, cout), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, SliceTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);

  torch::Tensor a = torch::randn({8, 3, 28, 28}, torch::requires_grad(false));
  torch::Tensor h_a = a.to(torch::kHABANA);
  int64_t dim = 1;
  int64_t start_index = 0;
  int64_t end = 8;
  int64_t step = 1;

  Tensor h_out = torch::slice(h_a, dim, start_index, end, step);

  std::vector<HbLazyTensor> tensors = {GetHbLazyTensor(h_out)};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  auto h_cout = h_out.to(torch::kCPU);
  auto cout = torch::slice(a, dim, start_index, end, step);

  EXPECT_EQ(allclose(h_cout, cout), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, ViewExecute) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto input_tensor =
      torch::arange(480, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape({10, 3, 4, 4}); // nchw
  torch::Tensor tHabanain = input_tensor.to(torch::kHABANA);
  c10::IntArrayRef new_size = {-1, 48};
  auto result = torch::_unsafe_view(tHabanain, new_size);
  auto hl_result = std::make_shared<HbLazyTensor>(GetHbLazyTensor(result));
  auto ir_value = hl_result->CurrentIrValue();
  std::vector<HbLazyTensor> tensors = {*hl_result};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});
  at::Tensor result_lazy = result.to(torch::kCPU);
  unsetenv("PT_HPU_LAZY_MODE");
  auto result_cpu = torch::_unsafe_view(input_tensor, new_size);
  EXPECT_EQ(allclose(result_lazy, result_cpu, 0.01, 0.01), true);
}

TEST_F(LazyTensorShapeKernelTest, TransposeTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor hOut = torch::transpose(hA, 1, 0);
  torch::Tensor Out = torch::transpose(A, 1, 0);

  EXPECT_EQ(allclose(hOut.to(torch::kCPU), Out), true);
  unsetenv("PT_HPU_LAZY_MODE");
}