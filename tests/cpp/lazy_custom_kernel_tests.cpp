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

class LazyCustomKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    setenv("PT_HPU_LAZY_MODE", "1", 1);
  }

  void TearDown() override {
    unsetenv("PT_HPU_LAZY_MODE");
  }
};

TEST_F(LazyCustomKernelTest, OptSgdCustomOp) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1});
  auto lr = torch::tensor({0.01});
  auto valid_cnt = torch::tensor({2});
  auto hgrad = grad.to(torch::kHABANA);
  auto hwts = wts.to(torch::kHABANA);
  auto hmoments = moments.to(torch::kHABANA);
  auto hindices = indices.to(torch::kHABANA);
  auto hlr = lr.to(torch::kHABANA);
  auto hvalid_cnt = valid_cnt.to(torch::kHABANA);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt, 0.1, false);
  auto I1 = torch::relu(out1);
  auto I2 = torch::relu(out2);

  auto hl_weight = GetHbLazyTensor(out1);
  auto hl_moment = GetHbLazyTensor(out2);
  std::vector<HbLazyTensor> tensors{hl_weight, hl_moment};
  std::vector<int> indices1{0, 1};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  std::vector<at::Tensor> input_list{
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(
      po_data.post_order,
      stack,
      po_data.inputs,
      po_data.outputs,
      po_data.post_order_nodes_hash);

  torch::jit::testing::FileCheck()
      .check("prim::Constant[value=0]")
      ->check("prim::Constant[value=0.10000000149011612]")
      ->check_count("habanaOptimizerSparseSgd", 1)
      ->run(*hlexec->get_graph());
}

TEST_F(LazyCustomKernelTest, OptAdagradCustomOp) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1});
  auto lr = torch::tensor({0.01});
  auto valid_cnt = torch::tensor({2});
  auto hgrad = grad.to(torch::kHABANA);
  auto hwts = wts.to(torch::kHABANA);
  auto hmoments = moments.to(torch::kHABANA);
  auto hindices = indices.to(torch::kHABANA);
  auto hlr = lr.to(torch::kHABANA);
  auto hvalid_cnt = valid_cnt.to(torch::kHABANA);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt);
  auto I1 = torch::relu(out1);
  auto I2 = torch::relu(out2);

  auto hl_weight = GetHbLazyTensor(out1);
  auto hl_moment = GetHbLazyTensor(out2);
  std::vector<HbLazyTensor> tensors{hl_weight, hl_moment};
  std::vector<int> indices1{0, 1};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  std::vector<at::Tensor> input_list{
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(
      po_data.post_order,
      stack,
      po_data.inputs,
      po_data.outputs,
      po_data.post_order_nodes_hash);

  torch::jit::testing::FileCheck()
      .check_count("habanaOptimizerSparseAdagrad", 1)
      ->run(*hlexec->get_graph());
}