/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/mixture_of_experts.h"

namespace habana {

MixtureOfExperts::MixtureOfExperts(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, "moe", scalar_type, {0}, {}, {}, false) {}

MixtureOfExpertsFusedWeights::MixtureOfExpertsFusedWeights(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(device_id, "moe", scalar_type, {0}, {}, {}, false) {}

static const std::map<c10::string_view, MoeActivationMode_t> activationModeMap =
    {{"gelu", MoeActivationMode_t::MOE_ACTIVATION_MODE_GELU},
     {"relu", MoeActivationMode_t::MOE_ACTIVATION_MODE_RELU},
     {"silu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SILU}};

void MixtureOfExperts::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto hidden_states = stack.at(0).toTensor();
  auto numExperts = stack.at(3).toTensorList().size();

  std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};
  for (size_t i = 3; i < 3 + numExperts * 3; i++) {
    inputs.push_back(syn_in(i));
  }

  const bool permuted_weights = stack.at(6).toBool();
  const auto activation_mode = stack.at(7).to<c10::string_view>();
  auto activationIterator = activationModeMap.find(activation_mode);
  TORCH_CHECK(
      activationIterator != activationModeMap.end(),
      "Activation \"",
      activation_mode,
      "\" not found among MoeActivationMode_t enum values.")

  ns_MoeKernel::ParamsV2 params;
  params.experts.activation = activationIterator->second;
  params.router.experts_min = stack.at(8).toScalar().toInt();
  params.router.experts_max = stack.at(9).toScalar().toInt();
  params.flags = permuted_weights ? MoeFlags_t::MOE_FLAGS_PERMUTED_WEIGHTS : 0;

  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe", hidden_states.scalar_type()),
       std::move(inputs),
       {{hidden_states.sizes(), hidden_states.scalar_type(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(moe_result[0]);
}

void MixtureOfExpertsFusedWeights::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto hidden_states = stack.at(0).toTensor();
  auto numExperts = stack.at(3).toTensorList().size();

  std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};
  for (size_t i = 3; i < 3 + numExperts * 2; i++) {
    inputs.push_back(syn_in(i));
  }

  const bool permuted_weights = stack.at(5).toBool();
  const auto activation_mode = stack.at(6).to<c10::string_view>();

  auto activationIterator = activationModeMap.find(activation_mode);
  TORCH_CHECK(
      activationIterator != activationModeMap.end(),
      "Activation \"",
      activation_mode,
      "\" not found among MoeActivationMode_t enum values.")

  ns_MoeKernel::ParamsV2 params;
  params.experts.activation = activationIterator->second;
  params.router.experts_min = stack.at(7).toScalar().toInt();
  params.router.experts_max = stack.at(8).toScalar().toInt();
  params.flags = permuted_weights ? MoeFlags_t::MOE_FLAGS_PERMUTED_WEIGHTS : 0;
  params.flags |= MoeFlags_t::MOE_FLAGS_FUSED_GEMM;

  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe", hidden_states.scalar_type()),
       std::move(inputs),
       {{hidden_states.sizes(), hidden_states.scalar_type(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(moe_result[0]);
}

} // namespace habana

static const auto& MoEKernelRegistry = habana::KernelRegistry().add(
    "hpu::mixture_of_experts",
    KERNEL_FN_GLOBAL(habana::MixtureOfExperts));

static const auto& MoEFusedWeightsKernelRegistry = habana::KernelRegistry().add(
    "hpu::mixture_of_experts.fused_weights",
    KERNEL_FN_GLOBAL(habana::MixtureOfExpertsFusedWeights));