/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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
#include "common_functions_custom_kernel_tests.h"
#include <gtest/gtest.h>
#include "common_functions_helpers.h"
#include "habana_kernels/lazy_kernels_declarations.h"
#include "habana_kernels/wrap_kernels_declarations.h"

void runResourceApplyMomentumOptTest(
    int num_params,
    int M,
    int N,
    double momentum,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> params_momentum_buf_list;
    std::vector<TensorAndView> dp_list;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto params_in = torch::randn({M, N});
    dump_tensor<float>(
        "params_in[" + std::to_string(i) + "]", params_in, verbose);
    PushBackHpuAndCpuTensors(
        params_in, hpu, cpu, &Data::params_momentum_buf_list, use_views);

    auto momentum_in = torch::randn({M, N});
    dump_tensor<float>(
        "momentum_in[" + std::to_string(i) + "]", momentum_in, verbose);
    PushBackHpuAndCpuTensors(
        momentum_in, hpu, cpu, &Data::params_momentum_buf_list, use_views);

    auto dp_in = torch::randn({M, N});
    dump_tensor<float>("dp_in[" + std::to_string(i) + "]", dp_in, verbose);
    PushBackHpuAndCpuTensors(dp_in, hpu, cpu, &Data::dp_list, use_views);
  }

  auto params_momentum_buf_list =
      TensorAndViewVecToViewVec(hpu.params_momentum_buf_list);
  auto dp_list = TensorAndViewVecToViewVec(hpu.dp_list);

  optimizer_resource_apply_momentum_hpu_wrap(
      params_momentum_buf_list, dp_list, momentum);

  // CPU calculations
  for (auto i = 0; i < num_params; i++) {
    const auto i2 = 2 * i;
    const auto i2p1 = i2 + 1;

    cpu.params_momentum_buf_list[i2p1].t.mul_(momentum).sub_(cpu.dp_list[i].t);
    cpu.params_momentum_buf_list[i2].t.add_(
        cpu.params_momentum_buf_list[i2p1].t);
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "params_list",
        std::make_pair(&Data::params_momentum_buf_list, 2 * i),
        "momentum_list",
        std::make_pair(&Data::params_momentum_buf_list, 2 * i + 1),
        "dp_list",
        &Data::dp_list);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}

void runLarsOptTest(
    int num_params,
    int M,
    int N,
    const std::vector<int64_t>& skip_masks,
    double eeta,
    double weight_decay,
    double eps,
    double lr,
    bool params_zero,
    bool grads_zero,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> params;
    std::vector<TensorAndView> grads;
  } cpu, hpu;

  std::vector<long> shape =
      (N > 1) ? std::vector<long>{M, N} : std::vector<long>{M};

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto params_in = params_zero ? torch::zeros(shape) : torch::randn(shape);
    dump_tensor<float>(
        "params_in[" + std::to_string(i) + "]", params_in, verbose);
    PushBackHpuAndCpuTensors(params_in, hpu, cpu, &Data::params, use_views);

    auto grads_in = grads_zero ? torch::zeros(shape) : torch::randn(shape);
    dump_tensor<float>(
        "grads_in[" + std::to_string(i) + "]", grads_in, verbose);
    PushBackHpuAndCpuTensors(grads_in, hpu, cpu, &Data::grads, use_views);
  }

  auto params = TensorAndViewVecToViewVec(hpu.params);
  auto grads = TensorAndViewVecToViewVec(hpu.grads);

  // auto lr_t = torch::full({1}, lr).to("hpu");

  optimizer_lars_hpu_wrap(
      params, grads, skip_masks, eeta, weight_decay, eps, lr);

  // CPU calculations
  for (auto i = 0; i < num_params; i++) {
    if (!skip_masks[i]) {
      cpu.grads[i].t.mul_(lr);
    } else {
      auto params_norm = cpu.params[i].t.square().sum().sqrt();
      auto grads_norm = cpu.grads[i].t.square().sum().sqrt();
      auto params_norm_positive = params_norm.greater(0.0);
      auto grads_norm_positive = grads_norm.greater(0.0);
      auto nominator = params_norm.mul(eeta);
      auto denominator = params_norm.mul(weight_decay).add(eps).add(grads_norm);
      auto division = nominator.div(denominator);
      auto selection = torch::where(
          grads_norm_positive,
          torch::where(params_norm_positive, division, 1.0),
          1.0);
      cpu.grads[i].t = cpu.params[i]
                           .t.mul(weight_decay)
                           .add(cpu.grads[i].t)
                           .mul(selection.mul(lr));
    }
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "params",
        &Data::params,
        "grads",
        &Data::grads);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}

void runLambPhase1OptimizerTest(
    int num_params,
    int M,
    int N,
    double weight_decay,
    int bias_correction,
    int step,
    int grad_averaging,
    bool with_view) {
  torch::manual_seed(0);
  const bool verbose = false;

  struct Data {
    std::vector<TensorAndView> grads_vec;
    std::vector<TensorAndView> weights_vec;
    std::vector<TensorAndView> exp_avg_vec;
    std::vector<TensorAndView> exp_avg_sq_vec;
    std::vector<TensorAndView> weight_norms_vec;
    std::vector<TensorAndView> adam_norms_vec;
    std::vector<TensorAndView> adam_steps_vec;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    auto grad = torch::randn({M, N});
    dump_tensor<float>("grad_in[" + std::to_string(i) + "]", grad, verbose);
    PushBackHpuAndCpuTensors(grad, hpu, cpu, &Data::grads_vec, with_view);

    auto weight = torch::randn({M, N});
    dump_tensor<float>("weight_in[" + std::to_string(i) + "]", weight, verbose);
    PushBackHpuAndCpuTensors(weight, hpu, cpu, &Data::weights_vec, with_view);

    auto exp_avg = torch::rand({M, N});
    dump_tensor<float>(
        "exp_avg_in[" + std::to_string(i) + "]", exp_avg, verbose);
    PushBackHpuAndCpuTensors(exp_avg, hpu, cpu, &Data::exp_avg_vec, with_view);

    auto exp_avg_sq = torch::rand({M, N});
    dump_tensor<float>(
        "exp_avg_sq_in[" + std::to_string(i) + "]", exp_avg_sq, verbose);
    PushBackHpuAndCpuTensors(
        exp_avg_sq, hpu, cpu, &Data::exp_avg_sq_vec, with_view);

    auto adam_norm = torch::zeros({1});
    dump_tensor<float>(
        "adam_norm_in[" + std::to_string(i) + "]", adam_norm, verbose);
    PushBackHpuAndCpuTensors(
        adam_norm, hpu, cpu, &Data::adam_norms_vec, with_view);

    auto weight_norm = torch::zeros({1});
    dump_tensor<float>(
        "weight_norm_in[" + std::to_string(i) + "]", weight_norm, verbose);
    PushBackHpuAndCpuTensors(
        weight_norm, hpu, cpu, &Data::weight_norms_vec, with_view);

    auto adam_step = torch::zeros({M, N});
    dump_tensor<float>(
        "adam_step_in[" + std::to_string(i) + "]", adam_step, verbose);
    PushBackHpuAndCpuTensors(
        adam_step, hpu, cpu, &Data::adam_steps_vec, with_view);
  }

  auto grads = TensorAndViewVecToViewVec(hpu.grads_vec);
  auto weights = TensorAndViewVecToViewVec(hpu.weights_vec);
  auto exp_avg = TensorAndViewVecToViewVec(hpu.exp_avg_vec);
  auto exp_avg_sq = TensorAndViewVecToViewVec(hpu.exp_avg_sq_vec);
  auto weight_norms = TensorAndViewVecToViewVec(hpu.weight_norms_vec);
  auto adam_norms = TensorAndViewVecToViewVec(hpu.adam_norms_vec);
  auto adam_steps = TensorAndViewVecToViewVec(hpu.adam_steps_vec);
  torch::Tensor cpu_clip_global_grad_norm = torch::ones({1});
  torch::Tensor hpu_clip_global_grad_norm =
      cpu_clip_global_grad_norm.to(torch::kHPU);

  float beta1 = 0.9;
  float beta2 = 0.999;
  float eps = 1e-6;

  // CPU calculation
  float beta3 = grad_averaging != 0 ? 1.0 - beta1 : 1.0;
  float bias_correction1 =
      bias_correction != 0 ? 1.0 - std::pow(beta1, step) : 1.0;
  float bias_correction2 =
      bias_correction != 0 ? 1.0 - std::pow(beta2, step) : 1.0;

  auto bias_correction1_t = torch::tensor(bias_correction1).to(torch::kHPU);
  auto bias_correction2_t = torch::tensor(bias_correction2).to(torch::kHPU);

  habana_lazy::optimizer_lamb_phase1(
      grads,
      weights,
      exp_avg,
      exp_avg_sq,
      weight_norms,
      adam_norms,
      adam_steps,
      hpu_clip_global_grad_norm,
      grad_averaging,
      beta1,
      beta2,
      eps,
      bias_correction1_t,
      bias_correction2_t,
      weight_decay);

  for (int i = 0; i < num_params; i++) {
    auto& grad = cpu.grads_vec[i].t.div_(cpu_clip_global_grad_norm);
    cpu.exp_avg_vec[i].t.mul_(beta1).add_(grad, beta3);
    cpu.exp_avg_sq_vec[i].t.mul_(beta2).addcmul_(grad, grad, 1 - beta2);

    auto exp_avg = cpu.exp_avg_vec[i].t.div(bias_correction1);
    auto exp_avg_sq = cpu.exp_avg_sq_vec[i].t.div(bias_correction2);

    cpu.weight_norms_vec[i].t = cpu.weights_vec[i].t.norm();

    cpu.adam_steps_vec[i].t = exp_avg.div(exp_avg_sq.sqrt().add_(eps));
    if (weight_decay != 0) {
      cpu.adam_steps_vec[i].t.add_(cpu.weights_vec[i].t, weight_decay);
    }

    cpu.adam_norms_vec[i].t = cpu.adam_steps_vec[i].t.norm();
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    equal &= CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        1e-06,
        1e-06,
        "exp_avg_vec",
        &Data::exp_avg_vec,
        "exp_avg_sq_vec",
        &Data::exp_avg_sq_vec,
        "weight_norms_vec",
        &Data::weight_norms_vec,
        "adam_norms_vec",
        &Data::adam_norms_vec,
        "adam_steps_vec",
        &Data::adam_steps_vec);
  }
  EXPECT_TRUE(equal);
}

void runLambPhase2OptimizerTest(
    int num_params,
    int M,
    int N,
    const double weight_decay,
    const bool use_lamb,
    const bool with_view) {
  torch::manual_seed(0);
  const bool verbose = false;
  float neg_step = -0.1;

  struct Data {
    std::vector<TensorAndView> weights_vec;
    std::vector<TensorAndView> adam_norms_vec;
    std::vector<TensorAndView> weight_norms_vec;
    std::vector<TensorAndView> adam_steps_vec;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    auto weight = torch::randn({M, N});
    dump_tensor<float>("weight_in[" + std::to_string(i) + "]", weight, verbose);
    PushBackHpuAndCpuTensors(weight, hpu, cpu, &Data::weights_vec, with_view);

    auto adam_norm = torch::rand({1});
    dump_tensor<float>(
        "adam_norm_in[" + std::to_string(i) + "]", adam_norm, verbose);
    PushBackHpuAndCpuTensors(adam_norm, hpu, cpu, &Data::adam_norms_vec, false);

    auto weight_norm = torch::rand({1});
    dump_tensor<float>(
        "weight_norm_in[" + std::to_string(i) + "]", weight_norm, verbose);
    PushBackHpuAndCpuTensors(
        weight_norm, hpu, cpu, &Data::weight_norms_vec, false);

    auto adam_step = torch::randn({M, N});
    dump_tensor<float>(
        "adam_step_in[" + std::to_string(i) + "]", adam_step, verbose);
    PushBackHpuAndCpuTensors(
        adam_step, hpu, cpu, &Data::adam_steps_vec, with_view);
  }

  auto weights = TensorAndViewVecToViewVec(hpu.weights_vec);
  auto adam_norms = TensorAndViewVecToViewVec(hpu.adam_norms_vec);
  auto weight_norms = TensorAndViewVecToViewVec(hpu.weight_norms_vec);
  auto adam_steps = TensorAndViewVecToViewVec(hpu.adam_steps_vec);

  auto tensor_step = torch::tensor(neg_step).to(torch::kHPU);
  habana_lazy::optimizer_lamb_phase2(
      weights,
      adam_norms,
      weight_norms,
      adam_steps,
      tensor_step,
      weight_decay,
      use_lamb);

  // CPU calculations
  for (int i = 0; i < num_params; i++) {
    torch::Tensor trust_ratio = torch::ones(1);
    if ((weight_decay != 0 || use_lamb) &&
        (cpu.adam_norms_vec[i].t[0].item<float>() > 0) &&
        (cpu.weight_norms_vec[i].t[0].item<float>() > 0)) {
      trust_ratio = cpu.weight_norms_vec[i].t / cpu.adam_norms_vec[i].t;
    }
    cpu.adam_steps_vec[i].t *= neg_step * trust_ratio;
    cpu.weights_vec[i].t =
        torch::add(cpu.weights_vec[i].t, cpu.adam_steps_vec[i].t, 1.0);
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    equal &= CompareFewTensors<float>(
        i, hpu, cpu, verbose, 1e-06, 1e-06, "weights_vec", &Data::weights_vec);
  }
  EXPECT_TRUE(equal);
}

void runEmaOptTest(
    int num_params,
    int M,
    int N,
    double decay_val,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> model_inputs;
    std::vector<TensorAndView> updated_ema;
    std::vector<TensorAndView> decay;
  } cpu, hpu;

  auto decay_in = torch::full({1}, decay_val);
  dump_tensor<float>("decay_in", decay_in, verbose);
  PushBackHpuAndCpuTensors(decay_in, hpu, cpu, &Data::decay, false);

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto model_inputs_in = torch::randn({M, N});
    dump_tensor<float>(
        "model_inputs_in[" + std::to_string(i) + "]", model_inputs_in, verbose);
    PushBackHpuAndCpuTensors(
        model_inputs_in, hpu, cpu, &Data::model_inputs, use_views);

    auto updated_ema_in = torch::randn({M, N});
    dump_tensor<float>(
        "updated_ema_in[" + std::to_string(i) + "]", updated_ema_in, verbose);
    PushBackHpuAndCpuTensors(
        updated_ema_in, hpu, cpu, &Data::updated_ema, use_views);
  }

  auto model_inputs = TensorAndViewVecToViewVec(hpu.model_inputs);
  auto updated_ema = TensorAndViewVecToViewVec(hpu.updated_ema);
  auto decay = TensorAndViewVecToViewVec(hpu.decay)[0];

  optimizer_ema_hpu_wrap(model_inputs, updated_ema, decay);

  // CPU calculations
  auto one_minus_decay = 1.0 - cpu.decay[0].t;
  for (auto i = 0; i < num_params; i++) {
    cpu.updated_ema[i].t.mul_(cpu.decay[0].t);
    cpu.updated_ema[i].t.add_(cpu.model_inputs[i].t.mul(one_minus_decay));
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "model_inputs",
        &Data::model_inputs,
        "updated_ema",
        &Data::updated_ema);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}

void runAdamwOptTest(
    int num_params,
    int M,
    int N,
    double weight_decay,
    bool enable_views) {
  const bool verbose = false;

  torch::manual_seed(0);

  struct Data {
    std::vector<TensorAndView> grad_vec;
    std::vector<TensorAndView> wt_vec;
    std::vector<TensorAndView> exp_avg_vec;
    std::vector<TensorAndView> exp_avg_sq_vec;
  } cpu, hpu;

  for (auto i = 0; i < num_params; ++i) {
    bool use_views = enable_views && (i == num_params / 2);

    auto t_in = torch::randn({M, N});
    dump_tensor<float>("wt_in[" + std::to_string(i) + "]", t_in, verbose);
    PushBackHpuAndCpuTensors(t_in, hpu, cpu, &Data::grad_vec, use_views);

    auto ones = torch::ones_like(t_in);
    PushBackHpuAndCpuTensors(ones, hpu, cpu, &Data::wt_vec, use_views);

    auto zeros1 = torch::zeros_like(t_in);
    PushBackHpuAndCpuTensors(zeros1, hpu, cpu, &Data::exp_avg_vec, use_views);

    auto zeros2 = torch::zeros_like(t_in);
    PushBackHpuAndCpuTensors(
        zeros2, hpu, cpu, &Data::exp_avg_sq_vec, use_views);
  }

  auto gradients = TensorAndViewVecToViewVec(hpu.grad_vec);
  auto weights = TensorAndViewVecToViewVec(hpu.wt_vec);
  auto exp_avg = TensorAndViewVecToViewVec(hpu.exp_avg_vec);
  auto exp_avg_sq = TensorAndViewVecToViewVec(hpu.exp_avg_sq_vec);

  auto lr = 0.1;
  auto neg_step_t = torch::tensor({-lr}).to(torch::kHPU);
  auto beta1 = 0.5;
  auto beta2 = 0.5;
  auto epsilon = 1e-3;
  auto step = 0;

  optimizer_adamw_hpu_wrap(
      gradients,
      weights,
      exp_avg,
      exp_avg_sq,
      neg_step_t,
      beta1,
      beta2,
      epsilon,
      weight_decay);

  // CPU calculations
  auto step_size = lr;
  for (auto i = 0; i < num_params; i++) {
    cpu.exp_avg_vec[i].t.mul_(beta1);
    cpu.exp_avg_vec[i].t.add_(cpu.grad_vec[i].t, (1.0 - beta1));

    cpu.exp_avg_sq_vec[i].t.mul_(beta2);
    cpu.exp_avg_sq_vec[i].t.addcmul_(
        cpu.grad_vec[i].t, cpu.grad_vec[i].t, (1.0 - beta2));

    auto denom = cpu.exp_avg_sq_vec[i].t.sqrt().add_(epsilon);
    auto ratio = torch::div(cpu.exp_avg_vec[i].t, denom);
    auto scaled_ratio = torch::mul(ratio, step_size);

    cpu.wt_vec[i].t.mul_(weight_decay);
    cpu.wt_vec[i].t.sub_(scaled_ratio);
  }

  bool equal = true;
  for (auto i = 0; i < num_params; i++) {
    bool equal1 = CompareFewTensors<float>(
        i,
        hpu,
        cpu,
        verbose,
        0.001,
        0.001,
        "grad_vec",
        &Data::grad_vec,
        "wt_vec",
        &Data::wt_vec);
    // Don't shorten to equal = equal && CompareFewTensors(...) as we want
    // CompareFewTensors() is executed even if equal is false beforehand, for
    // logging purpose.
    equal = equal && equal1;
  }
  EXPECT_TRUE(equal);
}
