/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <gtest/gtest.h>
#include <tests/cpp/habana_lazy_test_infra.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>
#include <stdexcept>
#include "common_functions_custom_kernel_tests.h"
#include "common_functions_helpers.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"
#include "habana_kernels_ver/wrap_kernels_declarations.h"
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
      hlgradients, hlweights, hlmoments, hepoch_num, hlr, 0.1, 0.1, 0.1, false);

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

  TensorList hlgradients(hgrad);
  TensorList hlweights(hwts);
  TensorList hlmoments(hmoments);
  auto hwts_before = torch::clone(hwts);

  torch::Tensor out1, out2;
  optimizer_sgd_momentum_hpu_wrap(
      hlgradients, hlweights, hlmoments, hepoch_num, hlr, 0.1, 0.1, 0.1, false);

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

  TensorList hlgradients(hgrad);
  TensorList hlweights(hwts);
  TensorList hlmoments(hmoments);
  auto hwts_before = torch::clone(hwts);

  torch::Tensor out1, out2;
  optimizer_sgd_momentum_hpu_wrap(
      hlgradients, hlweights, hlmoments, hepoch_num, hlr, 0.1, 0.1, 0.1, false);

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

TEST_F(LazyCustomKernelTest, AdamwOptTest) {
  torch::manual_seed(0);
  int num_params = 2;
  int M = 4;
  int N = 4;

  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> wt_vec;
  std::vector<torch::Tensor> exp_avg_vec;
  std::vector<torch::Tensor> exp_avg_sq_vec;

  std::vector<torch::Tensor> grad_vec_cpu;
  std::vector<torch::Tensor> wt_vec_cpu;
  std::vector<torch::Tensor> exp_avg_vec_cpu;
  std::vector<torch::Tensor> exp_avg_sq_vec_cpu;

  auto t_in = torch::randn({M, N});
  for (auto i = 0; i < num_params; i++) {
    auto t = t_in.to(torch::kHPU);
    grad_vec_cpu.push_back(t_in);
    auto tH = t.to(torch::kHPU);
    grad_vec.push_back(tH);
    wt_vec_cpu.push_back(torch::ones_like(t_in).view({M, N}));
    auto tH_w = torch::ones_like(t_in).to(torch::kHPU);
    wt_vec.push_back(tH_w.view({M, N}));
    exp_avg_vec_cpu.push_back(torch::zeros_like(t_in));
    auto tH_ea = torch::zeros_like(t_in).to(torch::kHPU);
    exp_avg_vec.push_back(tH_ea);
    exp_avg_sq_vec_cpu.push_back(torch::zeros_like(t_in));
    auto tH_ea_sq = torch::zeros_like(t_in).to(torch::kHPU);
    exp_avg_sq_vec.push_back(tH_ea_sq);
  }

  TensorList gradients(grad_vec);
  TensorList weights(wt_vec);
  TensorList exp_avg(exp_avg_vec);
  TensorList exp_avg_sq(exp_avg_sq_vec);

  auto lr = 0.1;
  auto neg_step_t = torch::tensor({-lr}).to(torch::kHPU);
  auto beta1 = 0.5;
  auto beta2 = 0.5;
  auto epsilon = 1e-3;
  auto step = 0;
  auto bias_correction = false;
  auto modified_weight_decay = 0.99; // 1-lr*wd
  optimizer_adamw_hpu_wrap(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      lr,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      modified_weight_decay);

  HbLazyTensor::StepMarker({});

  // CPU calculations
  auto step_size = lr;
  if (bias_correction) {
    auto bias_correction1 = 1.0 - std::pow(beta1, step);
    auto bias_correction2 = 1.0 - std::pow(beta2, step);
    step_size = step_size * std::sqrt(bias_correction2) / bias_correction1;
  }
  /*  This are the operations we need to perform per parameter
      exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
      exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
      denom = exp_avg_sq.sqrt().add_(group["eps"])
      ratio = torch.div(exp_avg, denom)
      scaled_ratio = torch.mul(ratio, step_size)
      p.data.sub_(scaled_ratio)
      if group["weight_decay"] > 0.0:
        p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])
  */

  for (auto i = 0; i < num_params; i++) {
    wt_vec_cpu[i].mul_(modified_weight_decay);
    exp_avg_vec_cpu[i].mul_(beta1);
    exp_avg_vec_cpu[i].add_(grad_vec_cpu[i], (1.0 - beta1));
    exp_avg_sq_vec_cpu[i].mul_(beta2);
    exp_avg_sq_vec_cpu[i].addcmul_(
        grad_vec_cpu[i], grad_vec_cpu[i], (1.0 - beta2));
    auto denom = exp_avg_sq_vec_cpu[i].sqrt().add_(epsilon);
    auto ratio = torch::div(exp_avg_vec_cpu[i], denom);
    auto scaled_ratio = torch::mul(ratio, step_size);
    wt_vec_cpu[i].sub_(scaled_ratio);
  }
  for (auto i = 0; i < num_params; i++) {
    bool equal =
        wt_vec_cpu[i].allclose(wt_vec[i].to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  }
}

RESOURCE_APPLY_MOMENTUM_OPT_TEST(LazyCustomKernelTest)

LARS_OPT_TEST(LazyCustomKernelTest)

TEST_F(LazyCustomKernelTest, EMATest) {
  torch::manual_seed(0);
  int num_params = 1;
  int M = 4;
  int N = 4;
  auto decay = 0.4567;
  auto d = torch::tensor({decay}).to(torch::kHPU);
  std::vector<torch::Tensor> model_inputs; // msd - prev val
  std::vector<torch::Tensor> updated_ema; // ema - value

  std::vector<torch::Tensor> model_inputs_cpu;
  std::vector<torch::Tensor> updated_ema_cpu;

  auto t_in = torch::randn({M, N});
  for (auto i = 0; i < num_params; i++) {
    model_inputs_cpu.push_back(t_in);
    auto t = t_in.to(torch::kHPU);
    model_inputs.push_back(t);

    updated_ema_cpu.push_back(torch::ones_like(t_in));
    auto tH_w = torch::ones_like(t_in).to(torch::kHPU);
    updated_ema.push_back(tH_w);
  }

  TensorList mdIn(model_inputs);
  TensorList updtEma(updated_ema);

  optimizer_ema_hpu_wrap(mdIn, updtEma, d);

  HbLazyTensor::StepMarker({});

  // CPU calculations
  // for k, v in self.ema.state_dict().items():
  // v *= d
  // v += (1. - d) * msd[k]

  // v = updated_ema
  // d = decay
  // msd = model.module.state_dict() - module_inputs

  for (auto i = 0; i < num_params; i++) {
    updated_ema_cpu[i] = updated_ema_cpu[i].mul(decay);
    model_inputs_cpu[i] = model_inputs_cpu[i].mul((1.0 - decay));
    updated_ema_cpu[i] = updated_ema_cpu[i].add_(model_inputs_cpu[i]);
  }

  for (auto i = 0; i < num_params; i++) {
    // std::cout << "CPU " << updated_ema_cpu[i].to(torch::kCPU) << "\n HPU " <<
    // updated_ema[i].to(torch::kCPU) << "i " << i << "\n";
    bool equal = updated_ema_cpu[i].allclose(
        updated_ema[i].to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  }
}

TEST_F(LazyCustomKernelTest, EMATest_1) {
  torch::manual_seed(0);
  int num_params = 5;
  int M = 4;
  int N = 4;
  auto decay = 0.9999;
  auto d = torch::tensor({decay}).to(torch::kHPU);

  std::vector<torch::Tensor> model_inputs; // msd - prev val
  std::vector<torch::Tensor> updated_ema; // ema - value

  std::vector<torch::Tensor> model_inputs_cpu;
  std::vector<torch::Tensor> updated_ema_cpu;

  auto t_in = torch::randn({M, N});
  for (auto i = 0; i < num_params; i++) {
    model_inputs_cpu.push_back(t_in);
    auto t = t_in.to(torch::kHPU);
    model_inputs.push_back(t);

    updated_ema_cpu.push_back(torch::ones_like(t_in));
    auto tH_w = torch::ones_like(t_in).to(torch::kHPU);
    updated_ema.push_back(tH_w);
  }

  TensorList mdIn(model_inputs);
  TensorList updtEma(updated_ema);

  optimizer_ema_hpu_wrap(mdIn, updtEma, d);

  HbLazyTensor::StepMarker({});

  // CPU calculations
  // for k, v in self.ema.state_dict().items():
  // v *= d
  // v += (1. - d) * msd[k]

  // v = updated_ema
  // d = decay
  // msd = model.module.state_dict() - module_inputs

  for (auto i = 0; i < num_params; i++) {
    updated_ema_cpu[i] = updated_ema_cpu[i].mul(decay);
    model_inputs_cpu[i] = model_inputs_cpu[i].mul((1.0 - decay));
    updated_ema_cpu[i] = updated_ema_cpu[i].add_(model_inputs_cpu[i]);
  }

  for (auto i = 0; i < num_params; i++) {
    // std::cout << "CPU " << updated_ema_cpu[i].to(torch::kCPU) << "\n HPU " <<
    // updated_ema[i].to(torch::kCPU) << "i " << i << "\n";
    bool equal = updated_ema_cpu[i].allclose(
        updated_ema[i].to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  }
}

TEST_F(LazyCustomKernelTest, EMATest_WtView) {
  torch::manual_seed(0);
  int num_params = 1;
  int M = 4;
  int N = 4;
  auto decay = 0.4567;
  auto d = torch::tensor({decay}).to(torch::kHPU);
  std::vector<torch::Tensor> model_inputs; // msd - prev val
  std::vector<torch::Tensor> updated_ema; // ema - value

  std::vector<torch::Tensor> model_inputs_cpu;
  std::vector<torch::Tensor> updated_ema_cpu;

  auto t_in = torch::randn({M, N});
  for (auto i = 0; i < num_params; i++) {
    model_inputs_cpu.push_back(t_in);
    auto t = t_in.to(torch::kHPU);
    model_inputs.push_back(t);

    updated_ema_cpu.push_back(torch::ones_like(t_in.view({M, N})));
    auto tH_w = torch::ones_like(t_in).to(torch::kHPU);
    tH_w = tH_w.view({M, N});
    updated_ema.push_back(tH_w);
  }

  TensorList mdIn(model_inputs);
  TensorList updtEma(updated_ema);

  optimizer_ema_hpu_wrap(mdIn, updtEma, d);

  HbLazyTensor::StepMarker({});

  for (auto i = 0; i < num_params; i++) {
    updated_ema_cpu[i] = updated_ema_cpu[i].mul(decay);
    model_inputs_cpu[i] = model_inputs_cpu[i].mul((1.0 - decay));
    updated_ema_cpu[i] = updated_ema_cpu[i].add_(model_inputs_cpu[i]);
  }

  for (auto i = 0; i < num_params; i++) {
    // std::cout << "CPU " << updated_ema_cpu[i].to(torch::kCPU) << "\n HPU " <<
    // updated_ema[i].to(torch::kCPU) << "i " << i << "\n";
    bool equal = updated_ema_cpu[i].allclose(
        updated_ema[i].to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  }
}

LAMB_PHASE2_OPT_TEST(LazyCustomKernelTest)