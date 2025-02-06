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

#include "generated/backend/mish.h"
#include "generated/backend/mish_backward.h"
namespace habana {
SharedMetaDataVector MishBackwardSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& grad = stack_tensor(stack, 0);
  const auto& self = stack_tensor(stack, 1);
  const auto dtype = grad.scalar_type();
  const auto rank = grad.dim();
  const auto selfRank = self.dim();

  SharedMetaDataVector metaVec;
  metaVec.reserve(7);
  SharedMetaTensor commonTensor = {rank, dtype};
  SharedMetaTensor selfTensor = {
      selfRank, dtype}; // precision type is "dtype" in all kernels so self
                        // tensor will be always casted to it

  for (const auto& op : {"sigmoid_fwd", "softplus_fwd"}) {
    SharedMetaData sharedMeta{op};
    sharedMeta.inputs_data = {selfTensor};
    sharedMeta.outputs_data = {commonTensor};
    metaVec.push_back(sharedMeta);
  }

  SharedMetaData tanhSharedMeta{"tanh_fwd"};
  tanhSharedMeta.inputs_data = {commonTensor};
  tanhSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(tanhSharedMeta);

  SharedMetaData multSelfSharedMeta{"mult"};
  multSelfSharedMeta.inputs_data = {selfTensor, commonTensor};
  multSelfSharedMeta.outputs_data = {commonTensor};
  metaVec.push_back(multSelfSharedMeta);

  for (const auto& op : {"mult", "sub", "add"}) {
    SharedMetaData sharedMeta{op};
    sharedMeta.inputs_data = {commonTensor, commonTensor};
    sharedMeta.outputs_data = {commonTensor};
    metaVec.push_back(sharedMeta);
  }

  return metaVec;
}

void Mishbackward::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto& outshape = stack_tensor(stack, 0).sizes();
  using namespace std::literals;
  auto sigmoid_out = BuildOp(
      graph,
      get_guid_with_precision("sigmoid_fwd"sv, ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType()}});

  auto softplus_out = BuildOp(
      graph,
      get_guid_with_precision("softplus_fwd"sv, ScalarType()),
      {syn_in(1)},
      {{outshape, ScalarType()}});

  auto tanh_out = BuildOp(
      graph,
      get_guid_with_precision("tanh_fwd"sv, ScalarType()),
      {softplus_out[0].get()},
      {{outshape, ScalarType()}});

  auto mul_out1 = BuildOp(
      graph,
      get_guid_with_precision("mult"sv, ScalarType()),
      {syn_in(1), sigmoid_out[0].get()},
      {{outshape, ScalarType()}});

  auto sq_out = BuildOp(
      graph,
      get_guid_with_precision("mult"sv, ScalarType()),
      {tanh_out[0].get(), tanh_out[0].get()},
      {{outshape, ScalarType()}});

  auto mul_out2 = BuildOp(
      graph,
      get_guid_with_precision("mult"sv, ScalarType()),
      {mul_out1[0].get(), sq_out[0].get()},
      {{outshape, ScalarType()}});

  auto sub_out = BuildOp(
      graph,
      get_guid_with_precision("sub"sv, ScalarType()),
      {mul_out1[0].get(), mul_out2[0].get()},
      {{outshape, ScalarType()}});

  auto add_out = BuildOp(
      graph,
      get_guid_with_precision("add"sv, ScalarType()),
      {tanh_out[0].get(), sub_out[0].get()},
      {{outshape, ScalarType()}});

  auto grad_input = BuildOp(
      graph,
      get_guid_with_precision("mult"sv, ScalarType()),
      {syn_in(0), add_out[0].get()},
      {{outshape, ScalarType(), 0}});
  syn_out(0) = std::move(grad_input[0]);
}
} // namespace habana
