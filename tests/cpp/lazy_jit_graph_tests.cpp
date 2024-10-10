#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
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
using namespace at;

class LazyJITTest : public habana_lazy_test::LazyTest {
 public:
  void SetUp() override {
    ForceMode(1);

    habana_lazy::StageSubmission::getInstance().resetCurrentAccumulatedOps();
  }
};
/**
 * Create a JIT graph and check the nodes created within it.
 */
TEST_F(LazyJITTest, CreateGraph) {
  torch::Tensor tensor_in1_cpu = torch::randn({2, 3});
  torch::Tensor tensor_in2_cpu = torch::randn({2, 3});

  torch::Tensor tensor_in1 = tensor_in1_cpu.to(torch::kHPU);
  torch::Tensor tensor_in2 = tensor_in2_cpu.to(torch::kHPU);

  Scalar alpha = 1.0f, beta = 1.0f;
  auto result = torch::add(tensor_in1, tensor_in2, alpha);
  auto result2 = torch::add(result, tensor_in2, beta);
  auto hl_result = SyncAndGetHbLazyTensor(result2);

  std::vector<HbLazyTensor> tensors = {hl_result};
  std::vector<int> indices = {0};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices);

  std::vector<at::Tensor> input_list{tensor_in1, tensor_in2};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=1.]")
      ->check_count("= hpu::add", 2)
      ->run(*hlexec->get_graph());

  auto result2_cpu = result2.to(torch::kCPU);
}

TEST_F(LazyJITTest, ExecuteGraph) {
  Scalar alpha = 10.0f;
  Tensor tensor_in1 = torch::rand({2, 3});
  Tensor tensor_in2 = torch::rand({2, 3});
  Tensor exp1 = add(tensor_in1, tensor_in2, alpha);
  Tensor exp2 = add(exp1, tensor_in1, alpha);

  torch::Tensor htensor_in1 = tensor_in1.to(torch::kHPU);
  torch::Tensor htensor_in2 = tensor_in2.to(torch::kHPU);
  auto result1 = torch::add(htensor_in1, htensor_in2, alpha);
  auto result2 = torch::add(result1, htensor_in1, alpha);

  Tensor out1 = result1.to(kCPU);
  Tensor out2 = result2.to(kCPU);

  EXPECT_EQ(allclose(out1, exp1), true);
  EXPECT_EQ(allclose(out2, exp2), true);
}

TEST_F(LazyJITTest, ExecuteGraphCustomSgd) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1}, torch::dtype(torch::kInt32));
  auto lr = torch::tensor({0.01}, torch::dtype(torch::kFloat));
  auto valid_cnt = torch::tensor({2}, torch::dtype(torch::kInt32));
  torch::Tensor out1_eager, out2_eager;
  torch::Tensor result1_eager, result2_eager;
  auto hwt_eager = wts.to(torch::kHPU);
  auto hmoment_eager = moments.to(torch::kHPU);

  auto eagerFn = [&]() {
    std::tie(out1_eager, out2_eager) =
        optimizer_sparse_sgd_with_valid_count_hpu_wrap(
            grad.to(torch::kHPU),
            hwt_eager,
            hmoment_eager,
            indices.to(torch::kHPU),
            lr.to(torch::kHPU),
            valid_cnt.to(torch::kHPU),
            0.1,
            false);
    result1_eager = out1_eager.to(kCPU);
    result2_eager = out2_eager.to(kCPU);
  };
  ExecuteEager(eagerFn);

  auto hgrad = grad.to(torch::kHPU);
  auto hwts = wts.to(torch::kHPU);
  auto hmoments = moments.to(torch::kHPU);
  auto hindices = indices.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);
  auto hvalid_cnt = valid_cnt.to(torch::kHPU);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt, 0.1, false);

  auto hl_result1 =
      std::make_shared<HbLazyTensor>(SyncAndGetHbLazyTensor(out1));
  auto hl_result2 =
      std::make_shared<HbLazyTensor>(SyncAndGetHbLazyTensor(out2));
  std::vector<HbLazyTensor> tensors = {*hl_result1, *hl_result2};
  HbLazyTensor::SyncTensorsGraph(&tensors);

  Tensor result1 = out1.to(kCPU);
  Tensor result2 = out2.to(kCPU);

  EXPECT_EQ(allclose(result1, result1_eager), true);
  EXPECT_EQ(allclose(result2, result2_eager), true);
}
