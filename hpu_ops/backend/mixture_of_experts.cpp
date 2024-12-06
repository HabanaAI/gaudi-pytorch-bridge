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

#include "hpu_ops/mixture_of_experts.h"

namespace habana {

MixtureOfExperts::MixtureOfExperts(
    int device_id,
    c10::ScalarType scalar_type,
    bool measurement_mode)
    : OpBackend(
          device_id,
          "moe",
          scalar_type,
          measurement_mode ? std::vector<int>{0, 0} : std::vector<int>{0},
          {},
          {},
          false) {}

static const std::map<c10::string_view, MoeActivationMode_t> activationModeMap =
    {{"gelu", MoeActivationMode_t::MOE_ACTIVATION_MODE_GELU},
     {"relu", MoeActivationMode_t::MOE_ACTIVATION_MODE_RELU},
     {"silu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SILU}};

std::shared_ptr<void> FillMixtureOfExpertsParams(
    const at::Stack& stack,
    size_t& size,
    const int permuted_weights_idx,
    const bool fused_gemm,
    const bool measurement_mode) {
  const auto permuted_weights = stack.at(permuted_weights_idx).toBool();
  const auto activation_mode =
      stack.at(permuted_weights_idx + 1).to<c10::string_view>();
  auto activationIterator = activationModeMap.find(activation_mode);
  TORCH_CHECK(
      activationIterator != activationModeMap.end(),
      "Activation \"",
      activation_mode,
      "\" not found among MoeActivationMode_t enum values.")

  PARAMS_STUB(ns_MoeKernel::ParamsV2);
  params->experts.activation = activationIterator->second;
  params->router.experts_min =
      stack.at(permuted_weights_idx + 2).toScalar().toInt();
  params->router.experts_max =
      stack.at(permuted_weights_idx + 3).toScalar().toInt();
  params->flags = permuted_weights ? MoeFlags_t::MOE_FLAGS_PERMUTED_WEIGHTS : 0;
  params->flags |= (fused_gemm ? MoeFlags_t::MOE_FLAGS_FUSED_GEMM : 0);
  params->flags |= (measurement_mode ? MoeFlags_t::MOE_FLAGS_CALC_AMAX : 0);
  return params;
}

OutputMetaDataVector MixtureOfExpertsMeta(
    const at::Stack& stack,
    bool measurement_mode) {
  const auto& self = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = self.sizes().vec();
  meta[0].dtype = self.scalar_type();
  if (measurement_mode) {
    auto numExperts = stack.at(3).toTensorList().size();
    OutputMetaData measurement_meta;
    measurement_meta.shape = {static_cast<int>(numExperts)};
    measurement_meta.dtype = self.scalar_type();
    meta.push_back(measurement_meta);
  }
  return meta;
}

void MixtureOfExperts::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const bool fused_weights = !stack.at(5).isTensorList();
  auto num_experts = stack.at(3).toTensorList().size();
  auto weights_per_expert = fused_weights ? 2 : 3;
  auto permute_weights_idx = fused_weights ? 5 : 6;
  auto measurement_mode =
      fused_weights ? stack.size() == 10 : stack.size() == 11;

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 3 + num_experts * weights_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, measurement_mode);
  auto meta = MixtureOfExpertsMeta(stack, measurement_mode);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0}};
  if (measurement_mode) {
    output_attrs.push_back({meta[1].shape, meta[1].dtype, 1});
  }

  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe", meta[0].dtype),
       std::move(inputs),
       output_attrs,
       params.get(),
       size});

  syn_out(0) = std::move(moe_result[0]);
  if (measurement_mode) {
    syn_out(1) = std::move(moe_result[1]);
  }
}

} // namespace habana

static const auto& MixtureOfExpertsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::mixture_of_experts", KERNEL_FN_ARG(MixtureOfExperts, false))
        .add(
            "hpu::mixture_of_experts.fused_weights",
            KERNEL_FN_ARG(MixtureOfExperts, false))
        .add(
            "hpu::mixture_of_experts.fp8_measurement",
            KERNEL_FN_ARG(MixtureOfExperts, true))
        .add(
            "hpu::mixture_of_experts.fp8_measurement_fused_weights",
            KERNEL_FN_ARG(MixtureOfExperts, true));
