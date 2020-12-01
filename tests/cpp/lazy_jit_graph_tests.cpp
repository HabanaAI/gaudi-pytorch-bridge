#include <gtest/gtest.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/ir_utils.h"

using namespace habana_lazy;

/**
 * Create a JIT graph and check the nodes created within it.
 */
TEST(LazyJITTest, CreateGraph) {
  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor tensor_in1_cpu = torch::randn({2, 3});
  torch::Tensor tensor_in2_cpu = torch::randn({2, 3});

  torch::Tensor tensor_in1 = tensor_in1_cpu.to(torch::kHABANA);
  torch::Tensor tensor_in2 = tensor_in2_cpu.to(torch::kHABANA);

  Scalar alpha = 4.0f, beta = 99.5f;
  auto result = add_tensor_hpu_wrap(tensor_in1, tensor_in2, alpha);
  // auto result = torch::add(tensor_in1, tensor_in2);

  // torch::Tensor tensor_in3 = torch::randn({2, 3}).to(torch::kHABANA);
  auto result2 = add_tensor_hpu_wrap(result, tensor_in2, beta);
  // auto result2 = torch::add(result, tensor_in2);
  auto hl_result = GetHbLazyTensor(result2);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->Create(po_data.post_order, po_data.inputs, po_data.outputs);

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=99.5]")
      ->check("prim::Constant[value=4.]")
      ->check_count("aten::add", 2)
      ->run(*hlexec->get_graph());

  auto result2_cpu = result2.to(torch::kCPU);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyJITTest, ExecuteGraph) {
  Scalar alpha = 10.0f;
  Tensor tensor_in1 = torch::rand({2, 3});
  Tensor tensor_in2 = torch::rand({2, 3});
  Tensor exp1 = add(tensor_in1, tensor_in2, alpha);
  Tensor exp2 = add(exp1, tensor_in1, alpha);

  setenv("PT_HPU_LAZY_MODE", "1", 1);
  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHABANA);
  torch::Tensor htensor_in2 = tensor_in2.to(torch::kHABANA);
  auto result1 = add_tensor_hpu_wrap(htensor_in1, htensor_in2, alpha);
  auto result2 = add_tensor_hpu_wrap(result1, htensor_in1, alpha);

  Tensor out1 = result1.to(kCPU);
  Tensor out2 = result2.to(kCPU);

  EXPECT_EQ(allclose(out1, exp1), true);
  EXPECT_EQ(allclose(out2, exp2), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyJITTest, DISABLED_ExecuteGraphCustomSgd) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1}, torch::dtype(torch::kInt32));
  auto lr = torch::tensor({0.01}, torch::dtype(torch::kFloat));
  auto valid_cnt = torch::tensor({2}, torch::dtype(torch::kInt32));
  torch::Tensor out1_eager, out2_eager;
  auto hwt_eager = wts.to(torch::kHABANA);
  auto hmoment_eager = moments.to(torch::kHABANA);

  std::tie(out1_eager, out2_eager) =
      optimizer_sparse_sgd_with_valid_count_hpu_wrap(
          grad.to(torch::kHABANA),
          hwt_eager,
          hmoment_eager,
          indices.to(torch::kHABANA),
          lr.to(torch::kHABANA),
          valid_cnt.to(torch::kHABANA),
          0.1,
          false);
  Tensor result1_eager = out1_eager.to(kCPU);
  Tensor result2_eager = out2_eager.to(kCPU);

  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto hgrad = grad.to(torch::kHABANA);
  auto hwts = wts.to(torch::kHABANA);
  auto hmoments = moments.to(torch::kHABANA);
  auto hindices = indices.to(torch::kHABANA);
  auto hlr = lr.to(torch::kHABANA);
  auto hvalid_cnt = valid_cnt.to(torch::kHABANA);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt, 0.1, false);

  auto hl_result1 = std::make_shared<HbLazyTensor>(GetHbLazyTensor(out1));
  auto hl_result2 = std::make_shared<HbLazyTensor>(GetHbLazyTensor(out2));
  std::vector<HbLazyTensor> tensors = {*hl_result1, *hl_result2};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  Tensor result1 = out1.to(kCPU);
  Tensor result2 = out2.to(kCPU);

  EXPECT_EQ(allclose(result1, result1_eager), true);
  EXPECT_EQ(allclose(result2, result2_eager), true);
  unsetenv("PT_HPU_LAZY_MODE");
}

TEST(LazyJITTest, ExecuteGraphCustomAdagrad) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1}, torch::dtype(torch::kInt32));
  auto lr = torch::tensor({0.01}, torch::dtype(torch::kFloat));
  auto valid_cnt = torch::tensor({2}, torch::dtype(torch::kInt32));
  auto hwt_eager = wts.to(torch::kHABANA);
  auto hmoment_eager = moments.to(torch::kHABANA);

  torch::Tensor out1_eager, out2_eager;
  std::tie(out1_eager, out2_eager) =
      optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
          grad.to(torch::kHABANA),
          hwt_eager,
          hmoment_eager,
          indices.to(torch::kHABANA),
          lr.to(torch::kHABANA),
          valid_cnt.to(torch::kHABANA));
  Tensor result1_eager = out1_eager.to(kCPU);
  Tensor result2_eager = out2_eager.to(kCPU);

  setenv("PT_HPU_LAZY_MODE", "1", 1);
  auto hgrad = grad.to(torch::kHABANA);
  auto hwts = wts.to(torch::kHABANA);
  auto hmoments = moments.to(torch::kHABANA);
  auto hindices = indices.to(torch::kHABANA);
  auto hlr = lr.to(torch::kHABANA);
  auto hvalid_cnt = valid_cnt.to(torch::kHABANA);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt);

  auto hl_result1 = std::make_shared<HbLazyTensor>(GetHbLazyTensor(out1));
  auto hl_result2 = std::make_shared<HbLazyTensor>(GetHbLazyTensor(out2));
  std::vector<HbLazyTensor> tensors = {*hl_result1, *hl_result2};
  HbLazyTensor::SyncTensorsGraph(&tensors, {});

  Tensor result1 = out1.to(kCPU);
  Tensor result2 = out2.to(kCPU);

  // NANs  are treated as equals to avoid random failures
  EXPECT_EQ(
      allclose(result1, result1_eager, 0.001, 0.001, /*equal_nan*/ true), true);
  EXPECT_EQ(allclose(result2, result2_eager), true);
  unsetenv("PT_HPU_LAZY_MODE");
}
