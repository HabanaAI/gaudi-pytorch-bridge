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

  matmul_test({2, 2, 2}, {2});
  matmul_test({2, 2, 2}, {2, 2, 2});
  matmul_test({12, 384, 1024}, {1024, 4096});
  matmul_test({12, 384, 768}, {768, 768});
  matmul_test({12, 384, 1024}, {1024, 1024});
  matmul_test({12, 16, 384, 64}, {12, 16, 64, 384});
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