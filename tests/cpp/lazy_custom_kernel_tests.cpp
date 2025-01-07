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
#include "common_functions_custom_kernel_tests.h"
#include "common_functions_helpers.h"
#include "generated/lazy/wrap_kernels_declarations.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_lazy/aten_lazy_bridge.h"
#include "habana_lazy/debug_utils.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir_utils.h"

#define HPU torch::kHPU
#define CPU torch::kCPU

using namespace habana_lazy;
using namespace at;

class LazyCustomKernelTest : public habana_lazy_test::LazyTest {};

TEST_F(LazyCustomKernelTest, OptSparseSgdCustomOp) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1});
  auto lr = torch::tensor({0.01});
  auto valid_cnt = torch::tensor({2});
  auto hgrad = grad.to(torch::kHPU);
  auto hwts = wts.to(torch::kHPU);
  auto hmoments = moments.to(torch::kHPU);
  auto hindices = indices.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);
  auto hvalid_cnt = valid_cnt.to(torch::kHPU);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_sgd_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt, 0.1, false);
  auto I1 = torch::relu(out1);
  auto I2 = torch::relu(out2);

  auto hl_weight = SyncAndGetHbLazyTensor(out1);
  auto hl_moment = SyncAndGetHbLazyTensor(out2);
  std::vector<HbLazyTensor> tensors{hl_weight, hl_moment};
  std::vector<int> indices1{0, 1};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  std::vector<at::Tensor> input_list{
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=0.10000000149011612]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check("= prim::Constant[value=0]")
      ->run(*hlexec->get_graph());

  torch::jit::testing::FileCheck()
      .check_count("= hpu::habanaOptimizerSparseSgd", 1)
      ->run(*hlexec->get_graph());
}

TEST_F(LazyCustomKernelTest, OptSgdMomentumCustomOp) {
  auto grad = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto wts = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto moments = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto epoch_num = torch::tensor({1});
  auto lr = torch::tensor({0.01});
  auto hmom = torch::tensor({0.1}).to(torch::kHPU);

  auto hgrad = grad.to(torch::kHPU);
  auto hwts = wts.to(torch::kHPU);
  auto hmoments = moments.to(torch::kHPU);
  auto hepoch_num = epoch_num.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);

  TensorList hlgradients(hgrad);
  TensorList hlweights(hwts);
  TensorList hlmoments(hmoments);

  torch::Tensor out1, out2;
  optimizer_sgd_momentum_hpu_wrap(
      hlgradients,
      hlweights,
      hlmoments,
      hepoch_num,
      hlr,
      hmom,
      0.1,
      0.1,
      false);

  auto in = torch::randn({64, 4, 28, 28}, torch::requires_grad());
  auto h_in = in.to(torch::kHPU);
  torch::Tensor result =
      torch::conv2d(h_in, hwts, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  // Sample optimizer+forward graph
  HbLazyTensor::StepMarker({});
}

TEST_F(LazyCustomKernelTest, OptSgdMomentumCustomOp_WtView) {
  auto grad = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto wts = torch::randn({5, 4, 3, 3}, torch::requires_grad(false))
                 .view({5, 4, 3, 3});
  auto moments = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto epoch_num = torch::tensor({1});
  auto lr = torch::tensor({0.01});

  auto hgrad = grad.to(torch::kHPU);
  auto hwts = wts.to(torch::kHPU).view({5, 4, 3, 3});
  auto hmoments = moments.to(torch::kHPU);
  auto hepoch_num = epoch_num.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);

  auto hmom = torch::tensor({0.1}).to(torch::kHPU);

  TensorList hlgradients(hgrad);
  TensorList hlweights(hwts);
  TensorList hlmoments(hmoments);
  auto hwts_before = torch::clone(hwts);

  torch::Tensor out1, out2;
  optimizer_sgd_momentum_hpu_wrap(
      hlgradients,
      hlweights,
      hlmoments,
      hepoch_num,
      hlr,
      hmom,
      0.1,
      0.1,
      false);

  auto in = torch::randn({64, 4, 28, 28}, torch::requires_grad());
  auto h_in = in.to(torch::kHPU);
  torch::Tensor result =
      torch::conv2d(h_in, hwts, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  // Sample optimizer+forward graph
  HbLazyTensor::StepMarker({});
  // std::cout << " orog wt after " << hwts.to(torch::kCPU) ;
  // std::cout << " orog wt after before " << hwts_before.to(torch::kCPU);

  // bool equal =
  //      hwts_before.allclose(hwts.to(torch::kCPU), 0.001, 0.001);
  // EXPECT_EQ(equal, false);
}

TEST_F(LazyCustomKernelTest, OptSgdMomentumCustomOp_Wt_Grad_View) {
  auto grad = torch::randn({5, 4, 3, 3}, torch::requires_grad(false))
                  .view({5, 4, 3, 3});
  auto wts = torch::randn({5, 4, 3, 3}, torch::requires_grad(false))
                 .view({5, 4, 3, 3});
  auto moments = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto epoch_num = torch::tensor({1});
  auto lr = torch::tensor({0.01});

  auto hgrad = grad.to(torch::kHPU).view({5, 4, 3, 3});
  auto hwts = wts.to(torch::kHPU).view({5, 4, 3, 3});
  auto hmoments = moments.to(torch::kHPU);
  auto hepoch_num = epoch_num.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);
  auto hmom = torch::tensor({0.1}).to(torch::kHPU);

  TensorList hlgradients(hgrad);
  TensorList hlweights(hwts);
  TensorList hlmoments(hmoments);
  auto hwts_before = torch::clone(hwts);

  torch::Tensor out1, out2;
  optimizer_sgd_momentum_hpu_wrap(
      hlgradients,
      hlweights,
      hlmoments,
      hepoch_num,
      hlr,
      hmom,
      0.1,
      0.1,
      false);

  auto in = torch::randn({64, 4, 28, 28}, torch::requires_grad());
  auto h_in = in.to(torch::kHPU);
  torch::Tensor result =
      torch::conv2d(h_in, hwts, {}, {1}, at::IntArrayRef{0}, {1}, 1);

  // Sample optimizer+forward graph
  HbLazyTensor::StepMarker({});
  // std::cout << " orog wt after " << hwts.to(torch::kCPU) ;
  // std::cout << " orog wt after before " << hwts_before.to(torch::kCPU);

  // bool equal =
  //      hwts_before.allclose(hwts.to(torch::kCPU), 0.001, 0.001);
  // EXPECT_EQ(equal, false);
}

TEST_F(LazyCustomKernelTest, OptAdagradCustomOp_WtView) {
  auto grad = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto wts = torch::randn({5, 4, 3, 3}, torch::requires_grad(false))
                 .view({5, 4, 3, 3});
  auto var = torch::randn({5, 4, 3, 3}, torch::requires_grad(false));
  auto epoch_num = torch::tensor({1});
  auto lr = torch::tensor({0.01});

  auto hgrad = grad.to(torch::kHPU);
  auto hwts = wts.to(torch::kHPU).view({5, 4, 3, 3});
  auto hvar = var.to(torch::kHPU);
  auto hepoch_num = epoch_num.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);

  TensorList hlgradients(hgrad);
  TensorList hlweights(hwts);
  TensorList hlvars(hvar);

  optimizer_adagrad_hpu_wrap(
      hlgradients, hlweights, hlvars, hepoch_num, hlr, 0.1, 0.1, 0.01);

  // Sample optimizer+forward graph
  HbLazyTensor::StepMarker({});
}

TEST_F(LazyCustomKernelTest, OptAdagradCustomOp) {
  auto grad = torch::randn({2, 2}, torch::requires_grad(false));
  auto wts = torch::randn({2, 2}, torch::requires_grad(false));
  auto moments = torch::randn({2, 2}, torch::requires_grad(false));
  auto indices = torch::tensor({0, 1});
  auto lr = torch::tensor({0.01});
  auto valid_cnt = torch::tensor({2});
  auto hgrad = grad.to(torch::kHPU);
  auto hwts = wts.to(torch::kHPU);
  auto hmoments = moments.to(torch::kHPU);
  auto hindices = indices.to(torch::kHPU);
  auto hlr = lr.to(torch::kHPU);
  auto hvalid_cnt = valid_cnt.to(torch::kHPU);
  torch::Tensor out1, out2;
  std::tie(out1, out2) = optimizer_sparse_adagrad_with_valid_count_hpu_wrap(
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt);
  auto I1 = torch::relu(out1);
  auto I2 = torch::relu(out2);

  auto hl_weight = SyncAndGetHbLazyTensor(out1);
  auto hl_moment = SyncAndGetHbLazyTensor(out2);
  std::vector<HbLazyTensor> tensors{hl_weight, hl_moment};
  std::vector<int> indices1{0, 1};
  auto po_data = HbLazyTensor::RunPostOrder(tensors, indices1);

  std::vector<at::Tensor> input_list{
      hgrad, hwts, hmoments, hindices, hlr, hvalid_cnt};

  auto stack = torch::jit::Stack(
      std::make_move_iterator(input_list.begin()),
      std::make_move_iterator(input_list.end()));

  exec::HlExec* hlexec = new exec::HlExec();
  hlexec->GetOrCreate(po_data, stack);

  torch::jit::testing::FileCheck()
      .check_count("= hpu::habanaOptimizerSparseAdagrad", 1)
      ->run(*hlexec->get_graph());
}

ADAMW_OPT_TEST(LazyCustomKernelTest, true)
EMA_OPT_TEST(LazyCustomKernelTest, true)
LAMB_PHASE1_OPT_TEST(LazyCustomKernelTest, true)
LAMB_PHASE2_OPT_TEST(LazyCustomKernelTest)
LARS_OPT_TEST(LazyCustomKernelTest, true)
RESOURCE_APPLY_MOMENTUM_OPT_TEST(LazyCustomKernelTest, true)
