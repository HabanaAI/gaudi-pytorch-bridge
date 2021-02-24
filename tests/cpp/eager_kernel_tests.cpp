#include <gtest/gtest.h>
#include <torch/torch.h>
#include <stdexcept>
#include "habana_kernels/linear_kernels.h"

#include "habana_kernels/eager_kernels_declarations.h"

TEST(EagerKernelTest, ReluTest) {
  torch::Tensor tensor = torch::randn({2, 3});
  torch::Tensor tHabana = tensor.to(torch::kHABANA);
  auto outHabana = torch::relu(tHabana);
  auto out = torch::relu(tensor);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(EagerKernelTest, AddTest) {
  torch::Tensor tensor = torch::randn({2, 3});
  torch::Tensor tHabana = tensor.to(torch::kHABANA);
  auto outHabana = torch::add(tHabana, 4.0);
  auto out = torch::add(tensor, 4.0);
  bool equal = out.allclose(outHabana.to(torch::kCPU), 0, 0);
  EXPECT_EQ(equal, true);
}

TEST(EagerKernelTest, MatMulTest) {
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    torch::Tensor tensor1 = torch::randn(size1);
    torch::Tensor tensor2 =
        torch::randn(size2); // torch::randn({2, 2}); d2 =1,2 tested and passing
    torch::Tensor ht1 = tensor1.to(torch::kHABANA);
    torch::Tensor ht2 = tensor2.to(torch::kHABANA);
    auto outHabana = matmul_hpu(ht1, ht2);
    auto out = torch::matmul(tensor1, tensor2);
    bool equal = out.allclose(outHabana.to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  };

  // Testing all configurations supported by CPU.
  // Do not delete from this list
  matmul_test({10}, {10});
  matmul_test({2, 10}, {10});
  matmul_test({10}, {10, 2});
  matmul_test({2, 10}, {10, 2});
  matmul_test({2, 3, 4}, {4});
  matmul_test({2, 3, 4}, {2, 4, 3});
  matmul_test({12, 20, 24}, {24, 20});
  matmul_test({12, 16, 20, 24}, {12, 16, 24, 20});
  matmul_test({3}, {2, 3, 4});
  matmul_test({3, 4}, {2, 4, 3});
  matmul_test({12, 16, 20, 24}, {16, 24, 20});
  matmul_test({16, 20, 24}, {12, 16, 24, 20});
}

TEST(EagerKernelTest, MatmulBackwardTest) {
  torch::manual_seed(0);
  auto matmul_test = [](c10::IntArrayRef size1, c10::IntArrayRef size2) {
    auto mat1 = torch::randn(size1);
    auto mat2 = torch::randn(size2);
    auto mat1_h = mat1.to(torch::kHABANA);
    auto mat2_h = mat2.to(torch::kHABANA);

    mat1.set_requires_grad(true);
    mat2.set_requires_grad(true);
    auto out = torch::matmul(mat1, mat2);

    auto grad_out = torch::ones_like(out);
    auto grad_out_h = grad_out.to(torch::kHABANA);
    out.backward(grad_out);
    auto grad_mat1 = mat1.grad();
    auto grad_mat2 = mat2.grad();

    torch::Tensor grad_mat1_h, grad_mat2_h;
    std::tie(grad_mat1_h, grad_mat2_h) =
        matmul_backward_hpu(grad_out_h, mat1_h, mat2_h);
    bool equal1 = grad_mat1.allclose(grad_mat1_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal1, true);
    bool equal2 = grad_mat2.allclose(grad_mat2_h.to(torch::kCPU), 0.01, 0.01);
    EXPECT_EQ(equal2, true);
  };
  matmul_test({2, 3, 4}, {4, 5});
  matmul_test({2, 3, 4}, {2, 4, 5});
  matmul_test({2, 3, 4}, {4});
  matmul_test({2, 2, 3, 4}, {2, 4, 3});
}

TEST(EagerKernelTest, AdamwOptTest) {
  torch::manual_seed(0);
  int num_params = 2;
  int M = 4;
  int N = 4;

  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> wt_vec;
  std::vector<torch::Tensor> exp_avg_vec;
  std::vector<torch::Tensor> exp_avg_sq_vec;
  auto t_in = torch::randn({M, N});
  for (auto i = 0; i < num_params; i++) {
    auto t = torch::randn({M, N}).to(torch::kHABANA);
    t.copy_(t_in);
    grad_vec.push_back(t);
    wt_vec.push_back(torch::ones_like(t));
    exp_avg_vec.push_back(torch::zeros_like(t));
    exp_avg_sq_vec.push_back(torch::zeros_like(t));
  }

  auto lr = 0.1;
  auto beta1 = 0.5;
  auto beta2 = 0.5;
  auto epsilon = 1e-3;
  auto step = 0;
  auto bias_correction = false;
  auto weight_decay = 0.0;
  optimizer_adamw_hpu(
      grad_vec,
      wt_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      lr,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay);

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
  std::vector<torch::Tensor> grad_vec_cpu;
  std::vector<torch::Tensor> wt_vec_cpu;
  std::vector<torch::Tensor> exp_avg_vec_cpu;
  std::vector<torch::Tensor> exp_avg_sq_vec_cpu;
  for (auto i = 0; i < num_params; i++) {
    auto t = t_in;
    grad_vec_cpu.push_back(t);
    wt_vec_cpu.push_back(torch::ones_like(t));
    exp_avg_vec_cpu.push_back(torch::zeros_like(t));
    exp_avg_sq_vec_cpu.push_back(torch::zeros_like(t));
  }
  for (auto i = 0; i < num_params; i++) {
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

TEST(EagerKernelCacheTest, AdamwOptTest) {
  torch::manual_seed(0);
  int num_params = 2;
  int M = 4;
  int N = 4;

  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> wt_vec;
  std::vector<torch::Tensor> exp_avg_vec;
  std::vector<torch::Tensor> exp_avg_sq_vec;

  auto t_in = torch::randn({M, N});
  for (auto i = 0; i < num_params; i++) {
    auto t = torch::randn({M, N}).to(torch::kHABANA);
    t.copy_(t_in);
    grad_vec.push_back(t);
    wt_vec.push_back(torch::ones_like(t));
    exp_avg_vec.push_back(torch::zeros_like(t));
    exp_avg_sq_vec.push_back(torch::zeros_like(t));
  }
  auto starting_lr = 0.1;
  auto lr = starting_lr;
  auto delta_lr = 0.00001;
  auto beta1 = 0.5;
  auto beta2 = 0.5;
  auto epsilon = 1e-3;
  auto step = 0;
  auto bias_correction = false;
  auto weight_decay = 0.0;
  for (int i = 0; i < 2; i++) {
    optimizer_adamw_hpu(
        grad_vec,
        wt_vec,
        exp_avg_vec,
        exp_avg_sq_vec,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
    lr -= delta_lr; // to check for cache hit with changing lr
  }

  // CPU calculations
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
  std::vector<torch::Tensor> grad_vec_cpu;
  std::vector<torch::Tensor> wt_vec_cpu;
  std::vector<torch::Tensor> exp_avg_vec_cpu;
  std::vector<torch::Tensor> exp_avg_sq_vec_cpu;
  for (auto i = 0; i < num_params; i++) {
    auto t = t_in;
    grad_vec_cpu.push_back(t);
    wt_vec_cpu.push_back(torch::ones_like(t));
    exp_avg_vec_cpu.push_back(torch::zeros_like(t));
    exp_avg_sq_vec_cpu.push_back(torch::zeros_like(t));
  }
  lr = starting_lr;
  for (int j = 0; j < 2; j++) {
    auto step_size = lr;
    if (bias_correction) {
      auto bias_correction1 = 1.0 - std::pow(beta1, step);
      auto bias_correction2 = 1.0 - std::pow(beta2, step);
      step_size = step_size * std::sqrt(bias_correction2) / bias_correction1;
    }

    for (auto i = 0; i < num_params; i++) {
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
    lr -= delta_lr;
  }
  for (auto i = 0; i < num_params; i++) {
    bool equal =
        wt_vec_cpu[i].allclose(wt_vec[i].to(torch::kCPU), 0.000001, 0.000001);
    EXPECT_EQ(equal, true);
  }
}

TEST(EagerKernelTest, FusedNormTest) {
  // torch::manual_seed(0);
  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> grad_vec_h;
  std::vector<torch::Tensor> grad_vec_norms;
  auto num_params = 4;
  // setup input grad tensor lists
  for (auto i = 0; i < num_params; i++) {
    auto t = torch::randn({2, 2});
    grad_vec.push_back(t);
    grad_vec_norms.push_back(torch::norm(t));
    auto tH = t.to(torch::kHABANA);
    grad_vec_h.push_back(tH);
  }
  // init max_norm
  torch::Tensor max_norm =
      torch::ones({1}, torch::TensorOptions().dtype(torch::kFloat32)) * 1.0;
  auto max_norm_hpu = max_norm.to(torch::kHABANA);
  // do hpu and cpu fused_norm calcs
  auto total_norm = fused_norm_hpu(grad_vec_h, max_norm_hpu, 2.0);
  auto total_norm_cpu = torch::norm(torch::stack(grad_vec_norms));
  // compare total_norm returned
  EXPECT_LT(
      std::abs(total_norm.item().toFloat() - total_norm_cpu.item().toFloat()),
      0.001);
  auto clip_coeff_cpu = max_norm / (total_norm_cpu + 1e-6);
  grad_vec_norms.clear();
  // do grad update on cpu since that is what hpu fused_norm_hpu does
  if (clip_coeff_cpu.item<float>() < 1.0) {
    for (auto i = 0; i < num_params; i++) {
      grad_vec.at(i) = grad_vec.at(i) * clip_coeff_cpu;
      grad_vec_norms.push_back(torch::norm(grad_vec.at(i)));
    }
  }
  // compare grad tensors after update
  for (auto i = 0; i < num_params; i++) {
    bool equal =
        grad_vec[i].allclose(grad_vec_h[i].to(torch::kCPU), 0.0001, 0.0001);
    EXPECT_EQ(equal, true);
  }

  // call fused norm kernels again to test caching in hpu
  total_norm = fused_norm_hpu(grad_vec_h, max_norm_hpu, 2.0);
  total_norm_cpu = torch::norm(torch::stack(grad_vec_norms));
  clip_coeff_cpu = max_norm / (total_norm_cpu + 1e-6);
  grad_vec_norms.clear();
  // do grad update on cpu since that is what hpu fused_norm_hpu does
  if (clip_coeff_cpu.item<float>() < 1.0) {
    for (auto i = 0; i < num_params; i++) {
      grad_vec.at(i) = grad_vec.at(i) * clip_coeff_cpu;
      grad_vec_norms.push_back(torch::norm(grad_vec.at(i)));
    }
  }
  // compare total_norm returned
  EXPECT_LT(
      std::abs(total_norm.item().toFloat() - total_norm_cpu.item().toFloat()),
      0.002);
  // compare grad tensors after update
  for (auto i = 0; i < num_params; i++) {
    bool equal =
        grad_vec[i].allclose(grad_vec_h[i].to(torch::kCPU), 0.001, 0.001);
    EXPECT_EQ(equal, true);
  }
}

TEST(EagerKernelTest, LambOptPh1Test) {
  torch::manual_seed(0);
  int num_params = 2;
  int M = 4;
  int N = 4;
  bool cache = true;

  std::vector<torch::Tensor> grad_vec_1;
  std::vector<torch::Tensor> wt_vec_1;
  std::vector<torch::Tensor> exp_avg_vec_1;
  std::vector<torch::Tensor> exp_avg_sq_vec_1;
  for (auto i = 0; i < num_params; i++) {
    auto t = torch::randn({M, N});
    auto t_hpu = t.to(torch::kHABANA);
    grad_vec_1.push_back(t_hpu);
    wt_vec_1.push_back(torch::ones_like(t_hpu));
    exp_avg_vec_1.push_back(torch::zeros_like(t_hpu));
    exp_avg_sq_vec_1.push_back(torch::zeros_like(t_hpu));
  }

  std::vector<torch::Tensor> grad_vec;
  std::vector<torch::Tensor> wt_vec;
  std::vector<torch::Tensor> exp_avg_vec;
  std::vector<torch::Tensor> exp_avg_sq_vec;
  std::vector<torch::Tensor> grad_vec_cpu;
  std::vector<torch::Tensor> wt_vec_cpu;
  std::vector<torch::Tensor> exp_avg_vec_cpu;
  std::vector<torch::Tensor> exp_avg_sq_vec_cpu;
  for (auto i = 0; i < num_params; i++) {
    auto t = torch::randn({M, N});
    auto t_hpu = t.to(torch::kHABANA);
    grad_vec.push_back(t_hpu);
    wt_vec.push_back(torch::ones_like(t_hpu));
    exp_avg_vec.push_back(torch::zeros_like(t_hpu));
    exp_avg_sq_vec.push_back(torch::zeros_like(t_hpu));
    grad_vec_cpu.push_back(t);
    wt_vec_cpu.push_back(torch::ones_like(t));
    exp_avg_vec_cpu.push_back(torch::zeros_like(t));
    exp_avg_sq_vec_cpu.push_back(torch::zeros_like(t));
  }

  auto clip_grad_norm = torch::tensor({0.4});
  auto lr = 0.1;
  auto beta1 = 0.5;
  auto beta2 = 0.5;
  auto epsilon = 1e-3;
  auto step = 1;
  auto bias_correction = true;
  auto weight_decay = 0.1;
  auto grad_averaging = 1;
  std::vector<torch::Tensor> weight_norm, adam_norm, adam_step;
  if (cache) {
    std::tie(weight_norm, adam_norm, adam_step) = optimizer_lamb_phase1_hpu(
        grad_vec_1,
        wt_vec_1,
        exp_avg_vec_1,
        exp_avg_sq_vec_1,
        clip_grad_norm.to(torch::kHABANA),
        grad_averaging,
        lr,
        beta1,
        beta2,
        epsilon,
        step,
        bias_correction,
        weight_decay);
  }
  std::tie(weight_norm, adam_norm, adam_step) = optimizer_lamb_phase1_hpu(
      grad_vec,
      wt_vec,
      exp_avg_vec,
      exp_avg_sq_vec,
      clip_grad_norm.to(torch::kHABANA),
      grad_averaging,
      lr,
      beta1,
      beta2,
      epsilon,
      step,
      bias_correction,
      weight_decay);
  float bias_correction1 = 1.0, bias_correction2 = 1.0;
  if (bias_correction) {
    bias_correction1 = 1.0 - std::pow(beta1, step);
    bias_correction2 = 1.0 - std::pow(beta2, step);
  }

  float beta3 = 1.0;
  if (grad_averaging) {
    beta3 = 1 - beta1;
  }

  std::vector<torch::Tensor> adam_step_cpu;
  std::vector<torch::Tensor> adam_norm_cpu;
  std::vector<torch::Tensor> wt_norm_cpu;
  for (auto i = 0; i < num_params; i++) {
    auto grad = grad_vec_cpu[i].div(clip_grad_norm);
    exp_avg_vec_cpu[i].mul_(beta1);
    exp_avg_vec_cpu[i].add_(grad, beta3);
    exp_avg_sq_vec_cpu[i].mul_(beta2);
    exp_avg_sq_vec_cpu[i].addcmul_(grad, grad, (1.0 - beta2));
    auto exp_avg = exp_avg_vec_cpu[i].div(bias_correction1);
    auto exp_avg_sq = exp_avg_sq_vec_cpu[i].div(bias_correction2);
    auto denom = exp_avg_sq.sqrt().add_(epsilon);
    auto adam_step = torch::div(exp_avg, denom);
    if (weight_decay)
      adam_step.add_(wt_vec_cpu[i], weight_decay);
    adam_step_cpu.push_back(adam_step);
    adam_norm_cpu.push_back(torch::norm(adam_step, 2.0));
    wt_norm_cpu.push_back(torch::norm(wt_vec_cpu[i], 2.0));
  }

  for (auto i = 0; i < num_params; i++) {
    bool equal =
        wt_norm_cpu[i].allclose(weight_norm[i].to(torch::kCPU), 0.001, 0.001);
    /*std::cout << wt_norm_cpu[i] << "\t" << weight_norm[i].to(torch::kCPU) <<
    "\n";*/
    EXPECT_EQ(equal, true);
    equal =
        adam_norm_cpu[i].allclose(adam_norm[i].to(torch::kCPU), 0.001, 0.001);
    /*std::cout << adam_norm_cpu[i] << "\t" << adam_norm[i].to(torch::kCPU) <<
    "\n";*/
    EXPECT_EQ(equal, true);
    equal =
        adam_step_cpu[i].allclose(adam_step[i].to(torch::kCPU), 0.001, 0.001);
    /*std::cout << adam_step_cpu[i] << "\t" << adam_step[i].to(torch::kCPU) <<
    "\n";*/
    EXPECT_EQ(equal, true);
  }
}