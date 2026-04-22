/**
 * Copyright (c) 2021-2026 Intel Corporation
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
#include "generated/backend/rrelu_with_noise.h"
#include "generated/backend/rrelu_with_noise_backward.h"
#include "habana_kernels/random_gen_kernels.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana {

OutputMetaDataVector RreluWithNoiseCguidMeta(const at::Stack& stack) {
  constexpr unsigned OUTPUTS_NUMBER = 2;
  auto input = stack_tensor(stack, 0);
  OutputMetaDataVector metaVec(OUTPUTS_NUMBER);

  for (unsigned i = 0; i < OUTPUTS_NUMBER; ++i) {
    metaVec[i].shape = input.sizes().vec();
    metaVec[i].dtype = input.scalar_type();
  }

  return metaVec;
}

SharedMetaDataVector RreluWithNoiseSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();

  SharedMetaDataVector out;
  out.reserve(1);
  auto& randUniformSharedMeta = out.emplace_back("rrelu_with_noise_cguid");
  randUniformSharedMeta.inputs_data.emplace_back(rank, dtype);
  randUniformSharedMeta.inputs_data.emplace_back(1, at::ScalarType::Int);
  randUniformSharedMeta.outputs_data.emplace_back(rank, dtype);
  randUniformSharedMeta.outputs_data.emplace_back(rank, dtype);

  return out;
}

using namespace std::literals;

bool is_rrelu_functional(const OpBackend& op) {
  return op.GetGuid().find("rrelu_with_noise_functional") != std::string::npos;
}

FillParamsT FillRreluWithNoiseParams(const at::Stack& stack) {
  PARAMS_STUB(ns_RreluWithNoiseKernel::Params);

  auto lower = stack.at(2).toScalar().to<float>();
  auto upper = stack.at(3).toScalar().to<float>();
  auto training = stack.at(4).toBool();

  params->lower = lower;
  params->upper = upper;
  params->training = training;

  return paramsT;
}

constexpr double eps = 1e-6;

void Rrelu_with_noise::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "Rrelu_with_noise::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto noiseIn = stackGetter.getNextInput<std::optional<TensorsPair>>();

  bool is_functional = is_rrelu_functional(*this);

  std::optional<int> noise_out_idx =
      is_functional ? std::optional<int>(1) : std::nullopt;

  auto meta = RreluWithNoiseCguidMeta(stack);
  auto params = FillRreluWithNoiseParams(stack);

  std::optional<synapse_helpers::tensor> noiseStorageOpt;
  auto [noise_in_storage_or_idx] = get_or_create_tensor<STORAGE_IDX>(
      *this,
      graph,
      noiseIn,
      input.pt_t.numel(),
      meta[1].dtype,
      0,
      noiseStorageOpt);

  std::vector<NodeAttr::NodeOutputAttr> node_output_attr = {
      {meta[0].shape, meta[0].dtype, 0}};
  if (noiseIn.has_value()) {
    node_output_attr.push_back(
        NodeAttr::NodeOutputAttr{
            meta[1].shape,
            meta[1].dtype,
            noise_out_idx,
            DATA_TENSOR,
            syn_type_na,
            noise_in_storage_or_idx});
  } else {
    node_output_attr.push_back({meta[1].shape, meta[1].dtype, 1});
  }

  auto output = BuildOp(
      graph,
      get_guid_with_precision("rrelu_with_noise_cguid"sv, meta[0].dtype),
      {syn_in(0), stack.at(5).isTensor() ? syn_in(2) : syn_seed()},
      node_output_attr,
      params.ptr(),
      params.size());

  syn_out(0) = std::move(output[0]);

  if (is_functional) {
    syn_out(1) = std::move(output[1]);
  } else {
    GetSynImplicitOutputs().emplace_back(
        PtInputIdxAndSynHelpTensor{
            1,
            std::move(output[1]),
            static_cast<size_t>(std::get<int>(noise_in_storage_or_idx))});
  }
}

SharedMetaDataVector RreluWithNoiseBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  auto grad = stack.at(0).toTensor();
  auto rank = grad.dim();
  auto resultType = grad.scalar_type();

  auto training = stack.at(5).toBool();
  auto lower = stack.at(3).toScalar().to<float>();
  auto upper = stack.at(4).toScalar().to<float>();
  const auto* guid = "leakyrelu_bwd";
  if (training && (upper - lower) > eps) {
    guid = "mult";
  }
  SharedMetaDataVector rreluBwdSharedMetaVec;
  rreluBwdSharedMetaVec.reserve(1);
  auto& rreluBwdSharedMeta = rreluBwdSharedMetaVec.emplace_back(guid);
  rreluBwdSharedMeta.inputs_data.emplace_back(rank, resultType);
  rreluBwdSharedMeta.inputs_data.emplace_back(rank, resultType);
  rreluBwdSharedMeta.outputs_data.emplace_back(rank, resultType);
  return rreluBwdSharedMetaVec;
}

void Rrelu_with_noise_bwd::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  auto training = stack.at(5).toBool();
  auto lower = stack.at(3).toScalar().to<float>();
  auto upper = stack.at(4).toScalar().to<float>();
  if (training && (upper - lower) > eps) {
    // grad_out * noise
    auto output = BuildOp(
        graph,
        get_guid_with_precision("mult"sv, ScalarType()),
        {syn_in(0), syn_in(2)},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(output[0]);
  } else {
    PARAMS_STUB(ns_LeakyReluKernel::Params);
    auto negative_slope = (lower + upper) / 2;
    params->alpha = negative_slope;
    auto output = BuildOp(
        graph,
        get_guid_with_precision("leakyrelu_bwd"sv, ScalarType()),
        {syn_in(0), syn_in(1)},
        {{outshape, ScalarType(), 0}},
        paramsT.ptr(),
        paramsT.size());
    syn_out(0) = std::move(output[0]);
  }
}
} // namespace habana
