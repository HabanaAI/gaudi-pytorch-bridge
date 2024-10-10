/******************************************************************************
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
#include "common_functions_norm_kernel_tests.h"
#include <torch/torch.h>
#include "common_functions_helpers.h"
#include "variable_get.h"

std::vector<AtTensorPair> native_layer_norm_test(
    NativeLayerNormTestMode mode,
    NativeLayerNormTestWeight weight,
    NativeLayerNormTestBias bias,
    c10::ScalarType dtype,
    int dsIterNo,
    int dsItersCount,
    bool verbose) {
  std::vector<AtTensorPair> result;
  result.reserve(mode == NativeLayerNormTestMode::FwdBwdAffine ? 6 : 3);

  std::vector<long> input_shape;
  std::vector<long> normalized_shape;

  int typicallyTen = 10;
  if (dsItersCount > 1) {
    typicallyTen += 2 * dsIterNo - dsItersCount + 1;
    std::cout << "Iteration Start -- " << dsIterNo << " ----" << std::endl;
  }

  switch (mode) {
    case NativeLayerNormTestMode::FwdBwdAffine:
      input_shape = {2, 1, 2, 4}; // nchw
      normalized_shape = {1, 2, 4};
      break;
    case NativeLayerNormTestMode::BackwardGal:
      input_shape = {2, 3, 4};
      normalized_shape = {4};
      break;
    default:
      input_shape = {typicallyTen, 1, 3, 4, 4}; // nchw
      normalized_shape = {1, 3, 4, 4};
      break;
  }

  auto input_num_samples =
      c10::multiply_integers(input_shape.begin(), input_shape.end());
  auto norm_num_samples =
      c10::multiply_integers(normalized_shape.begin(), normalized_shape.end());

  auto input_tensor_cpu =
      torch::arange(
          input_num_samples, torch::dtype(torch::kFloat).requires_grad(false))
          .reshape(input_shape)
          .to(dtype);
  torch::Tensor input_tensor_hpu = input_tensor_cpu.to(torch::kHPU);
  if (dtype != torch::kFloat32) {
    input_tensor_cpu = input_tensor_cpu.to(torch::kFloat32);
  }

  dump_tensor<float>("Input:", input_tensor_cpu, verbose);

  c10::optional<at::Tensor> weight_cpu_opt;
  c10::optional<at::Tensor> bias_cpu_opt;
  c10::optional<torch::Tensor> weight_hpu_opt;
  c10::optional<torch::Tensor> bias_hpu_opt;
  if (weight == NativeLayerNormTestWeight::Defined) {
    weight_cpu_opt =
        torch::arange(
            norm_num_samples, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape(normalized_shape)
            .to(dtype);
    weight_hpu_opt = weight_cpu_opt->to(torch::kHPU);
    if (dtype != torch::kFloat32) {
      weight_cpu_opt = weight_cpu_opt->to(torch::kFloat32);
    }

    dump_tensor<float>("Weight:", *weight_cpu_opt, verbose);
  }
  if (bias == NativeLayerNormTestBias::Defined) {
    bias_cpu_opt =
        torch::arange(
            norm_num_samples, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape(normalized_shape)
            .to(dtype);
    bias_hpu_opt = bias_cpu_opt->to(torch::kHPU);
    if (dtype != torch::kFloat32) {
      bias_cpu_opt = bias_cpu_opt->to(torch::kFloat32);
    }

    dump_tensor<float>("Bias:", *bias_cpu_opt, verbose);
  }

  switch (mode) {
    case NativeLayerNormTestMode::Backward:
    case NativeLayerNormTestMode::BackwardGal:
      break;
    default: {
      auto results_hpu = torch::native_layer_norm(
          input_tensor_hpu,
          normalized_shape,
          weight_hpu_opt,
          bias_hpu_opt,
          0.01);

      auto results_cpu = torch::native_layer_norm(
          input_tensor_cpu,
          normalized_shape,
          weight_cpu_opt,
          bias_cpu_opt,
          0.01);

      for (int i = 0; i < 3; ++i) {
        result.emplace_back();
        result.back().hpu =
            (VariableGet<0, 3>::get(i, results_hpu)).to(torch::kCPU);
        result.back().cpu = VariableGet<0, 3>::get(i, results_cpu);

        std::string label =
            std::string("Fwd output ") + std::to_string(i) + ':';
        dump_tensors<float>(
            label, result.back().hpu, result.back().cpu, verbose);
      }
    }
  }

  if (mode != NativeLayerNormTestMode::Forward) {
    auto grad_out_cpu =
        torch::arange(
            input_num_samples, torch::dtype(torch::kFloat).requires_grad(false))
            .reshape(input_shape)
            .to(dtype); // nchw
    torch::Tensor grad_out_hpu = grad_out_cpu.to(torch::kHPU);
    if (dtype != torch::kFloat32) {
      grad_out_cpu = grad_out_cpu.to(torch::kFloat32);
    }

    dump_tensor<float>("Grad_out:", grad_out_cpu, verbose);

    at::Tensor mean_cpu;
    at::Tensor rstd_cpu;
    torch::Tensor mean_hpu;
    torch::Tensor rstd_hpu;

    if (mode == NativeLayerNormTestMode::FwdBwdAffine) {
      mean_cpu = result[1].cpu;
      rstd_cpu = result[2].cpu;
      mean_hpu = result[1].hpu.to(torch::kHPU);
      rstd_hpu = result[2].hpu.to(torch::kHPU);
    } else {
      int64_t N = 0;
      std::array<int64_t, 2> mean_rstd_shape = {};
      if (mode == NativeLayerNormTestMode::BackwardGal) {
        N = input_shape[0] * input_shape[1];
        mean_rstd_shape = {input_shape[0], input_shape[1]};
      } else {
        N = input_shape[0];
        mean_rstd_shape = {N, 1};
      }

      mean_cpu =
          torch::arange(N, torch::dtype(torch::kFloat).requires_grad(false))
              .reshape(mean_rstd_shape)
              .to(dtype);
      mean_hpu = mean_cpu.to(torch::kHPU);
      if (dtype != torch::kFloat32) {
        mean_cpu = mean_cpu.to(torch::kFloat32);
      }

      rstd_cpu =
          torch::arange(N, torch::dtype(torch::kFloat).requires_grad(false))
              .reshape(mean_rstd_shape)
              .to(dtype);
      rstd_hpu = rstd_cpu.to(torch::kHPU);
      if (dtype != torch::kFloat32) {
        rstd_cpu = rstd_cpu.to(torch::kFloat32);
      }

      dump_tensor<float>("Mean:", mean_cpu, verbose);
      dump_tensor<float>("Rstd:", rstd_cpu, verbose);
    }

    if (!weight_cpu_opt) {
      // layer_norm_backward_cpu does not work for empty weight/bias optionals
      // The problem is that it creates empty tensors based
      // on empty optionals that is effectively creating empty tensors
      // without device causing error:
      // C++ exception with description "tensor does not have a device
      // Exception raised from device_default at
      // /npu-stack/pytorch-fork/c10/core/TensorImpl.h:1223
      weight_cpu_opt = torch::ones(
          normalized_shape, torch::dtype(torch::kFloat).requires_grad(false));
    }

    if (!bias_cpu_opt) {
      bias_cpu_opt = torch::zeros(
          normalized_shape, torch::dtype(torch::kFloat).requires_grad(false));
    }

    std::array<bool, 3> output_mask = {true, true, true};

    auto results_hpu = torch::native_layer_norm_backward(
        grad_out_hpu,
        input_tensor_hpu,
        normalized_shape,
        mean_hpu,
        rstd_hpu,
        weight_hpu_opt,
        bias_hpu_opt,
        output_mask);

    auto results_cpu = torch::native_layer_norm_backward(
        grad_out_cpu,
        input_tensor_cpu,
        normalized_shape,
        mean_cpu,
        rstd_cpu,
        weight_cpu_opt,
        bias_cpu_opt,
        output_mask);

    for (int i = 0; i < 3; ++i) {
      result.emplace_back();
      result.back().hpu =
          (VariableGet<0, 3>::get(i, results_hpu)).to(torch::kCPU);
      result.back().cpu = VariableGet<0, 3>::get(i, results_cpu);

      std::string label = std::string("Bwd output ") + std::to_string(i) + ':';
      dump_tensors<float>(label, result.back().hpu, result.back().cpu, verbose);
    }
  }

  if (dsItersCount > 1) {
    std::cout << "Iteration End -- " << dsIterNo << " ----" << std::endl;
  }

  return result;
}
