/**
 * Copyright (c) 2021-2025 Intel Corporation
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

#include "generated/backend/nan_to_num.h"
#include "hpu_ops/shared_meta_common.h"

namespace habana {
SharedMetaDataVector NanToNumSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto rank = self.dim();
  const auto dtype = self.scalar_type();
  const auto isIntegralType = c10::isIntegralType(dtype, true);
  const SharedMetaTensor commonTensor = {rank, dtype};
  if (isIntegralType) {
    SharedMetaData memcpySharedMeta{"memcpy"};
    memcpySharedMeta.inputs_data = {commonTensor};
    memcpySharedMeta.outputs_data = {commonTensor};
    return {memcpySharedMeta};
  }

  SharedMetaDataVector metaVec;
  metaVec.reserve(4);
  // Don't create a shared meta for a constant node, as it would disable SAG
  // flow. In fact, this node is always created with a shape {1}, which is
  // supported by SAG.

  auto isNanMetaVec = IsFiniteInfNanSharedMeta(stack, "isnan_fwd");
  metaVec.insert(
      std::end(metaVec), std::begin(isNanMetaVec), std::end(isNanMetaVec));

  auto isInfPosNegMetaVec = IsFiniteInfNanSharedMeta(stack, "isinf_fwd");
  metaVec.insert(
      std::end(metaVec),
      std::begin(isInfPosNegMetaVec),
      std::end(isInfPosNegMetaVec));

  SharedMetaTensor whereOutputTensor = {rank, c10::ScalarType::Bool};
  SharedMetaData whereNanSharedMeta{"where_fwd"};
  whereNanSharedMeta.inputs_data = {
      isNanMetaVec[0].outputs_data[0], commonTensor, commonTensor};
  whereNanSharedMeta.outputs_data = {whereOutputTensor};
  metaVec.push_back(whereNanSharedMeta);

  SharedMetaData wherePosNegSharedMeta{"where_fwd"};
  wherePosNegSharedMeta.inputs_data = {
      isInfPosNegMetaVec[0].outputs_data[0], commonTensor, whereOutputTensor};
  wherePosNegSharedMeta.outputs_data = {whereOutputTensor};
  metaVec.push_back(wherePosNegSharedMeta);

  return metaVec;
}

void NantoNum::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const at::Tensor self = stack_tensor(stack, 0);
  const auto outshape = stack_tensor(stack, 0).sizes();

  // If input is of integral dtype then copy it into output
  if (c10::isIntegralType(ScalarType(), true)) {
    auto copy =
        BuildOp(graph, "memcpy", {syn_in(0)}, {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(copy[0]);
  } else {
    auto nan_constant = stack.at(1).isNone() ? 0.0 : stack.at(1).toDouble();

    auto const_nan = ConstantHelper(graph, nan_constant, ScalarType());

    c10::Scalar max_value;

    if (self.dtype() == c10::ScalarType::BFloat16) {
      max_value = std::numeric_limits<c10::BFloat16>::max();
    } else {
      max_value = std::numeric_limits<float>::max();
    }

    auto posinf_constant =
        stack.at(2).isNone() ? max_value : stack.at(2).toDouble();

    auto const_posinf = ConstantHelper(graph, posinf_constant, ScalarType());

    c10::Scalar min_value;

    if (self.dtype() == c10::ScalarType::BFloat16) {
      min_value = std::numeric_limits<c10::BFloat16>::lowest();
    } else {
      min_value = std::numeric_limits<float>::lowest();
    }

    auto neginf_constant =
        stack.at(3).isNone() ? min_value : stack.at(3).toDouble();

    auto const_neginf = ConstantHelper(graph, neginf_constant, ScalarType());

    const at::ScalarType& result_type = c10::ScalarType::Bool;

    using namespace std::literals;
    auto nan_mask = BuildOp(
        graph,
        get_guid_with_precision("isnan_fwd"sv, ScalarType()),
        {syn_in(0)},
        {{outshape, result_type}});

    ns_IsInfKernel::Params params_pos{};
    params_pos.detect_positive = 1;

    auto posinf_mask = BuildOp(
        graph,
        get_guid_with_precision("isinf_fwd"sv, ScalarType()),
        {syn_in(0)},
        {{outshape, result_type}},
        &params_pos,
        sizeof(params_pos));

    ns_IsInfKernel::Params params_neg{};
    params_neg.detect_negative = 1;

    auto neginf_mask = BuildOp(
        graph,
        get_guid_with_precision("isinf_fwd"sv, ScalarType()),
        {syn_in(0)},
        {{outshape, result_type}},
        &params_neg,
        sizeof(params_neg));

    auto where_nan = BuildOp(
        graph,
        get_guid_with_precision("where_fwd"sv, ScalarType()),
        {nan_mask[0].get(), const_nan.get(), syn_in(0)},
        {{outshape, ScalarType()}});

    auto where_pos = BuildOp(
        graph,
        get_guid_with_precision("where_fwd"sv, ScalarType()),
        {posinf_mask[0].get(), const_posinf.get(), where_nan[0].get()},
        {{outshape, ScalarType()}});

    auto where_neg = BuildOp(
        graph,
        get_guid_with_precision("where_fwd"sv, ScalarType()),
        {neginf_mask[0].get(), const_neginf.get(), where_pos[0].get()},
        {{outshape, ScalarType(), 0}});
    syn_out(0) = std::move(where_neg[0]);
  }
}
} // namespace habana
