/**
 * Copyright (c) 2026 Intel Corporation
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

#include "generated/backend/causal_conv1d_update.h"

#include <ATen/core/stack.h>

#include <string_view>
#include <vector>

namespace habana {

FillParamsT FillCausalConv1dUpdateParams(const at::Stack& stack) {
  const bool activation = stack.at(4).toBool();
  const bool has_bias = !stack.at(3).isNone();

  PARAMS_STUB(ns_CausalConv1DUpdate::Params);
  params->activation = activation ? CAUSAL_CONV1D_ACTIVATION_SILU
                                  : CAUSAL_CONV1D_ACTIVATION_NONE;
  params->bias = has_bias;
  params->cache_seqlens = false;
  params->conv_state_indices = false;
  params->pad_slot_id = stack.at(5).toInt();
  return paramsT;
}

OutputMetaDataVector CausalConv1dUpdateMeta(const at::Stack& stack) {
  const auto& x = stack.at(0).toTensor();
  const auto& conv_state = stack.at(1).toTensor();

  OutputMetaDataVector metaVec(2);
  metaVec[0].shape = x.sizes().vec();
  metaVec[0].dtype = x.scalar_type();
  metaVec[1].shape = conv_state.sizes().vec();
  metaVec[1].dtype = conv_state.scalar_type();
  return metaVec;
}

void CausalConv1dUpdate::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  using namespace std::literals;

  StackGetter stackGetter(this, stack, "CausalConv1dUpdate::AddNode");
  const auto& x = stackGetter.getNextInput<TensorsPair>();
  const auto& conv_state = stackGetter.getNextInput<TensorsPair>();
  const auto& weight = stackGetter.getNextInput<TensorsPair>();
  const auto bias = stackGetter.getNextInput<std::optional<TensorsPair>>();
  const bool activation = stackGetter.getNextInput<bool>();
  stackGetter.getNextInput<long>();

  if (bias && activation) {
    SetGuid(get_guid_with_precision(
        "causal_conv1d_update_bias_silu"sv, ScalarType()));
  } else if (bias) {
    SetGuid(
        get_guid_with_precision("causal_conv1d_update_bias"sv, ScalarType()));
  } else if (activation) {
    SetGuid(
        get_guid_with_precision("causal_conv1d_update_silu"sv, ScalarType()));
  } else {
    SetGuid(get_guid_with_precision("causal_conv1d_update"sv, ScalarType()));
  }

  const auto params = FillParams(stack);
  const auto metas = OutputMeta(stack);

  std::vector<synTensor> syn_inputs{x.syn_t, conv_state.syn_t, weight.syn_t};
  if (bias) {
    syn_inputs.push_back(bias->syn_t);
  }

  auto outputs = BuildOp(
      graph,
      guid_,
      std::move(syn_inputs),
      {{metas[0].shape, metas[0].dtype, 0},
       {metas[1].shape, metas[1].dtype, 1}},
      params.ptr(),
      params.size());

  syn_out(0) = std::move(outputs[0]);
  syn_out(1) = std::move(outputs[1]);
}

} // namespace habana
