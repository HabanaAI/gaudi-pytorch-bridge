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

TEST_F(LazyTensorShapeKernelTest, PermuteTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor out = hA.permute({1, 0});

  torch::Tensor expected = torch::randn({3, 2});
  EXPECT_EQ(out.sizes(), expected.sizes());
  auto hl_result = GetHbLazyTensor(out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check("int[] = prim::Constant[value=[1, 0]]")
      ->check("Tensor = aten::permute")
      ->run(*exec.get_graph());
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST_F(LazyTensorShapeKernelTest, TTest) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor A = torch::randn({2, 3}, torch::requires_grad(false));
  torch::Tensor hA = A.to(torch::kHABANA);
  torch::Tensor out = torch::t(hA);

  torch::Tensor expected = torch::randn({3, 2});
  EXPECT_EQ(out.sizes(), expected.sizes());
  auto hl_result = GetHbLazyTensor(out);
  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  auto exec = habana_lazy::exec::HlExec();
  exec.Create(po_data.post_order, po_data.inputs, po_data.outputs);
  torch::jit::testing::FileCheck()
      .check("Tensor = aten::t")
      ->run(*exec.get_graph());
  unsetenv("PT_HPU_LAZY_MODE");
}