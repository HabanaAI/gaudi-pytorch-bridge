/******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

RMSNorm::RMSNorm(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "rms_norm_ex_fwd",
          scalar_type,
          {0, 0},
          {},
          {},
          false) {
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

  std::vector<synTensor> inputs = {data_in.syn_t, gamma.syn_t};

  std::vector<int64_t> inverse_root_mean_square_sizes{
      data_in.pt_t.sizes().vec()};
  inverse_root_mean_square_sizes.back() = 1;

  const auto data_in_dtype =
      (data_in.pt_t.scalar_type() != gamma.pt_t.scalar_type())
      ? c10::ScalarType::Float
      : data_in.pt_t.scalar_type();

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {data_in.pt_t.sizes(), data_in_dtype, 0},
      {inverse_root_mean_square_sizes, c10::ScalarType::Float, 1}};

  auto output = OpBackend::BuildNode(
      this, graph, {GetGuid(), inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]); // root_mean_square_norm
  syn_out(1) = std::move(output[1]); // inverse_root_mean_square
}

RMSNormBackward::RMSNormBackward(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "rms_norm_ex_bwd",
          scalar_type,
          {1, 2},
          {},
          {},
          false) {}

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

  ns_RmsNorm::ParamsV3 params{};
  params.useStages = use_stages;
  params.bwdMode = static_cast<RmsNormBwdMode_t>(bwd_mode);

  std::vector<synTensor> inputs = {
      grad_in.syn_t, data_in.syn_t, gamma.syn_t, inverse_rms.syn_t};

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {data_in.pt_t.sizes(), data_in.pt_t.scalar_type(), 0},
      {gamma.pt_t.sizes(), gamma.pt_t.scalar_type(), 1}};

  auto output = OpBackend::BuildNode(
      this, graph, {GetGuid(), inputs, output_attrs, &params, sizeof(params)});

  syn_out(0) = std::move(output[0]); // grad_out
  syn_out(1) = std::move(output[1]); // grad_gamma_full
}

} // namespace habana

static const auto& RMSNormKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::rms_norm", KERNEL_FN_GLOBAL(habana::RMSNorm))
        .add(
            "hpu::rms_norm_backward",
            KERNEL_FN_GLOBAL(habana::RMSNormBackward));
