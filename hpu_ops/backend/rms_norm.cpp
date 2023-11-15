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

#include "hpu_ops/rms_norm.h"
#include "backend/habana_device/hpu_cached_devices.h"

namespace habana {

sizes_vec RMSNormOutputShape(const at::Stack& stack) {
  auto data_in = stack_tensor(stack, 0);
  std::vector<int64_t> data_in_sizes = data_in.sizes().vec();

  std::vector<int64_t> inverse_root_mean_square_sizes{data_in_sizes};
  inverse_root_mean_square_sizes.back() = 1;

  return {data_in_sizes, inverse_root_mean_square_sizes};
}

static std::vector<int64_t> CalculateNewSizes(
    std::vector<int64_t> input_sizes) {
  TORCH_CHECK(
      input_sizes.size() >= 2, "Input sizes must be equal or greater than 2");

  std::vector<int64_t> input_new_sizes{input_sizes.at(0) * input_sizes.at(1)};
  for (size_t i = 2; i < input_sizes.size(); ++i)
    input_new_sizes.push_back(input_sizes.at(i));

  return input_new_sizes;
}

RMSNorm::RMSNorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "rms_norm", scalar_type, {0, 0}, {}, {}, false) {
  SetComputeOutputShapes(RMSNormOutputShape);
}

void RMSNorm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(stack, "RMSNorm::AddNode");
  auto data_in = getNextInput<TensorsPair>(stackGetter);
  auto gamma = getNextInput<TensorsPair>(stackGetter);
  auto epsilon = getNextInput<double>(stackGetter);

  ns_LayerNormKernel::Params params{};
  params.epsValid = true;
  params.eps = static_cast<float>(epsilon);

  auto data_in_sizes = data_in.pt_t.sizes().vec();
  auto data_in_new_sizes = CalculateNewSizes(data_in_sizes);

  auto data_in_reshaped = ReshapeHelper(
      graph, data_in.syn_t, data_in_new_sizes, data_in.pt_t.scalar_type());

  std::vector<synTensor> inputs = {data_in_reshaped.get(), gamma.syn_t};

  std::vector<int64_t> inverse_root_mean_square_new_sizes{data_in_new_sizes};
  inverse_root_mean_square_new_sizes.back() = 1;

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {data_in_new_sizes, data_in.pt_t.scalar_type()},
      {inverse_root_mean_square_new_sizes, c10::ScalarType::Float}};

  auto rms_norm_output = OpBackend::BuildNode(
      this, graph, {GetGuid(), inputs, output_attrs, &params, sizeof(params)});

  auto inverse_rms_output_org = data_in_sizes;
  inverse_rms_output_org.back() = 1;

  auto root_mean_square_norm_reshaped = ReshapeHelper(
      graph,
      rms_norm_output[0].get(),
      data_in_sizes,
      data_in.pt_t.scalar_type(),
      0);
  auto inverse_root_mean_square_reshaped = ReshapeHelper(
      graph,
      rms_norm_output[1].get(),
      inverse_rms_output_org,
      c10::ScalarType::Float,
      1);

  syn_out(0) = std::move(root_mean_square_norm_reshaped);
  syn_out(1) = std::move(inverse_root_mean_square_reshaped);
}

RMSNormBackward::RMSNormBackward(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "rms_norm_bwd", scalar_type, {1, 2}, {}, {}, false) {
}

void RMSNormBackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "RMSNormBackward::AddNode");
  auto grad_in = getNextInput<TensorsPair>(stackGetter);
  auto data_in = getNextInput<TensorsPair>(stackGetter);
  auto gamma = getNextInput<TensorsPair>(stackGetter);
  auto inverse_rms = getNextInput<TensorsPair>(stackGetter);
  auto use_stages = getNextInput<bool>(stackGetter);
  auto bwd_mode = getNextInput<int>(stackGetter);

  auto data_in_sizes = data_in.pt_t.sizes().vec();
  auto data_in_new_sizes = CalculateNewSizes(data_in_sizes);

  auto grad_in_sizes = grad_in.pt_t.sizes().vec();
  auto grad_in_new_sizes = CalculateNewSizes(grad_in_sizes);

  auto inverse_rms_sizes = inverse_rms.pt_t.sizes().vec();
  auto inverse_rms_new_sizes = CalculateNewSizes(inverse_rms_sizes);

  auto data_in_reshaped = ReshapeHelper(
      graph, data_in.syn_t, data_in_new_sizes, data_in.pt_t.scalar_type());

  auto grad_in_reshaped = ReshapeHelper(
      graph, grad_in.syn_t, grad_in_new_sizes, grad_in.pt_t.scalar_type());

  auto inverse_rms_reshaped = ReshapeHelper(
      graph,
      inverse_rms.syn_t,
      inverse_rms_new_sizes,
      inverse_rms.pt_t.scalar_type());

  std::vector<synTensor> inputs = {
      data_in_reshaped.get(),
      gamma.syn_t,
      grad_in_reshaped.get(),
      inverse_rms_reshaped.get()};

  if (use_stages) {
    int numTpc = 8;

    auto type{habana::HPURegistrar::get_device().type()};
    if (type == synDeviceGaudi2)
      numTpc = 24;
    else if (type == synDeviceGaudi3)
      numTpc = 64;

    int W = data_in.pt_t.sizes()[data_in.pt_t.sizes().size() - 2];
    int grad_gamma_partial_dim0 = std::min(numTpc, W);

    ns_RmsNorm::ParamsV2 params{};
    params.bwdStage = 1;
    params.bwdMode = static_cast<RmsNormBwdMode_t>(bwd_mode);

    std::vector<int64_t> grad_gamma_partial_sizes{
        grad_gamma_partial_dim0, data_in.pt_t.sizes().vec().back()};

    std::vector<NodeAttr::NodeOutputAttr> output_attrs_stage1 = {
        {data_in_new_sizes, data_in.pt_t.scalar_type()}, // grad_out
        {grad_gamma_partial_sizes,
         gamma.pt_t.scalar_type()}}; // grad_gamma_partial

    auto stage1 = OpBackend::BuildNode(
        this,
        graph,
        {GetGuid(), inputs, output_attrs_stage1, &params, sizeof(params)});

    auto stage1_data_in_reshaped = ReshapeHelper(
        graph, stage1[0].get(), data_in_sizes, data_in.pt_t.scalar_type(), 0);

    params.bwdStage = 2;

    auto stage2 = OpBackend::BuildNode(
        this,
        graph,
        {GetGuid(),
         {stage1[1].get()},
         {{gamma.pt_t.sizes(), gamma.pt_t.scalar_type(), 1}}, // grad_gamma_full
         &params,
         sizeof(params)});

    syn_out(0) = std::move(stage1_data_in_reshaped); // grad_out
    syn_out(1) = std::move(stage2[0]); // grad_gamma_full
  } else {
    auto rms_norm_dx_bwd = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("rms_norm_dx_bwd", ScalarType()),
         inputs,
         {{data_in_new_sizes, data_in.pt_t.scalar_type()}},
         nullptr,
         0});

    auto rms_norm_dx_bwd_grad_out_reshaped = ReshapeHelper(
        graph,
        rms_norm_dx_bwd[0].get(),
        grad_in_sizes,
        grad_in.pt_t.scalar_type(),
        0);

    auto rms_norm_dgamma_bwd = OpBackend::BuildNode(
        this,
        graph,
        {get_guid_with_precision("rms_norm_dgamma_bwd", ScalarType()),
         {data_in_reshaped.get(),
          grad_in_reshaped.get(),
          inverse_rms_reshaped.get()},
         {{gamma.pt_t.sizes(), gamma.pt_t.scalar_type(), 1}},
         nullptr,
         0});

    syn_out(0) = std::move(rms_norm_dx_bwd_grad_out_reshaped); // grad_out
    syn_out(1) = std::move(rms_norm_dgamma_bwd[0]); // grad_gamma_full
  }
}

} // namespace habana

static const auto& RMSNormKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::rms_norm", KERNEL_FN_GLOBAL(habana::RMSNorm))
        .add(
            "hpu::rms_norm_backward",
            KERNEL_FN_GLOBAL(habana::RMSNormBackward));
