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
#include <perf_lib_layer_params.h>
#include "generated/backend/native_group_norm.h"
#include "generated/backend/native_group_norm_backward.h"

namespace habana {

namespace sh = synapse_helpers;

sizes_vec NativeGroupNormFwdOutputShape(const at::Stack& stack) {
  const auto input_size = stack[0].toTensor().sizes().vec();
  const int N = stack[3].toInt();
  const int G = stack[6].toInt();

  return {input_size, {N, G}, {N, G}};
}

OutputMetaDataVector GroupNormFwdMeta(const at::Stack& stack) {
  constexpr unsigned OUTPUTS_NUMBER = 3;
  auto input = stack_tensor(stack, 0);
  auto shapes = NativeGroupNormFwdOutputShape(stack);
  OutputMetaDataVector metaVec(OUTPUTS_NUMBER);

  for (unsigned i = 0; i < OUTPUTS_NUMBER; ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = input.scalar_type();
  }

  return metaVec;
}

SharedMetaDataVector NativeGroupNormFwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  auto input = stack_tensor(stack, 0);
  auto rank = input.dim();
  auto dtype = input.scalar_type();
  auto N = stack.at(3).toInt();
  auto C = stack.at(4).toInt();
  auto HxW = stack.at(5).toInt();
  if (N * C * HxW == 0) {
    SharedMetaData memsetSharedMeta{"memset"};
    memsetSharedMeta.outputs_data.emplace_back(rank, dtype);
    SharedMetaData memsetMeanRstdSharedMeta{"memset"};
    memsetMeanRstdSharedMeta.outputs_data.emplace_back(2, dtype);

    return {memsetSharedMeta, memsetMeanRstdSharedMeta};
  }

  auto weight = stack.at(1).toOptional<at::Tensor>().value_or(at::Tensor());
  auto bias = stack.at(2).toOptional<at::Tensor>().value_or(at::Tensor());

  SharedMetaData nativeGroupNormSharedMeta{"native_group_norm_fwd"};
  nativeGroupNormSharedMeta.inputs_data.emplace_back(rank, dtype);
  if (weight.defined())
    nativeGroupNormSharedMeta.inputs_data.emplace_back(weight.dim(), dtype);
  else
    nativeGroupNormSharedMeta.inputs_data.push_back(
        createOptionalNotPresentSharedMetaTensor());

  if (bias.defined())
    nativeGroupNormSharedMeta.inputs_data.emplace_back(bias.dim(), dtype);
  nativeGroupNormSharedMeta.outputs_data = {
      {rank, dtype}, {2, dtype}, {2, dtype}};

  return {nativeGroupNormSharedMeta};
}

SharedMetaDataVector NativeGroupNormBwdSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& gradOut = stack_tensor(stack, 0);
  const auto& input = stack_tensor(stack, 1);
  const auto& mean = stack_tensor(stack, 2);
  const auto& rstd = stack_tensor(stack, 3);
  const auto weight =
      stack.at(4).toOptional<at::Tensor>().value_or(at::Tensor());

  const auto inputRank = input.dim();
  const auto inputDtype = input.scalar_type();

  SharedMetaData nativeSharedMeta{"native_group_norm_bwd"};

  if (weight.defined()) {
    nativeSharedMeta.inputs_data = {
        {gradOut.dim(), gradOut.scalar_type()},
        {inputRank, inputDtype},
        {mean.dim(), mean.scalar_type()},
        {rstd.dim(), rstd.scalar_type()},
        {weight.dim(), weight.scalar_type()}};
  } else {
    nativeSharedMeta.inputs_data = {
        {gradOut.dim(), gradOut.scalar_type()},
        {inputRank, inputDtype},
        {mean.dim(), mean.scalar_type()},
        {rstd.dim(), rstd.scalar_type()}};
  }

  nativeSharedMeta.outputs_data = {
      {inputRank, inputDtype},
      {{1}, inputDtype},
      {{1}, inputDtype},
  };

  return {nativeSharedMeta};
}

FillParamsT FillNativeGroupNormParams(const at::Stack& stack) {
  PARAMS_STUB(ns_NativeGroupNorm::Params);
  params->N = stack[3].toInt();
  params->G = stack[6].toInt();
  params->epsilon = stack[7].toDouble();

  return paramsT;
}

FillParamsT FillNativeGroupNormBwdParams(const at::Stack& stack) {
  PARAMS_STUB(ns_NativeGroupNorm::Params);
  params->N = stack[5].toInt();
  params->G = stack[8].toInt();

  return paramsT;
}

sizes_vec NativeGroupNormBwdOutputShape(const at::Stack& stack) {
  auto input = stack[1].toTensor();
  auto input_size = input.sizes().vec();

  int weight_size = stack[6].toInt();

  return {input_size, {weight_size}, {weight_size}};
}

OutputMetaDataVector GroupNormBwdMeta(const at::Stack& stack) {
  constexpr unsigned OUTPUTS_NUMBER = 3;
  auto input = stack_tensor(stack, 1);
  auto shapes = NativeGroupNormBwdOutputShape(stack);
  OutputMetaDataVector metaVec(OUTPUTS_NUMBER);

  for (unsigned i = 0; i < OUTPUTS_NUMBER; ++i) {
    metaVec[i].shape = shapes[i];
    metaVec[i].dtype = input.scalar_type();
  }

  return metaVec;
}

void NativeGroupNormFwd::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "NativeGroupNormFwd::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto weight = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto bias = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto metas = OutputMeta(stack);
  auto params = FillParams(stack);
  auto outputsNumber = metas.size();
  if (input.pt_t.numel() == 0) {
    for (unsigned i = 0; i < outputsNumber; i++) {
      auto output =
          BuildOp(graph, "memset", {}, {{metas[i].shape, metas[i].dtype, i}});
      syn_out(i) = std::move(output[0]);
    }
  } else {
    const auto rank = input.pt_t.dim();
    auto layout = [rank]() {
      if (rank == 3) {
        return synapse_helpers::layouts::SynapseLayoutFormat::WCN;
      } else if (rank == 4) {
        return synapse_helpers::layouts::SynapseLayoutFormat::WHCN;
      } else {
        return synapse_helpers::layouts::SynapseLayoutFormat::WHDCN;
      }
    }();

    SetSynapseLayouts(
        {layout,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
        {layout,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});

    auto outputs = BuildOp(
        graph,
        GetGuid(),
        {input.syn_t,
         weight.has_value() ? weight.value().syn_t : nullptr,
         bias.has_value() ? bias.value().syn_t : nullptr},
        {{metas[0].shape, metas[0].dtype, 0},
         {metas[1].shape, metas[1].dtype, 1},
         {metas[2].shape, metas[2].dtype, 2}},
        params.ptr(),
        params.size());
    for (unsigned i = 0; i < outputsNumber; i++) {
      syn_out(i) = std::move(outputs[i]);
    }
  }
}

void NativeGroupNormBwd::AddNode(sh::graph& graph, const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "NativeGroupNormBwdHabanaOperator::AddNode");
  auto grad_in = stackGetter.getNextInput<TensorsPair>();
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto mean = stackGetter.getNextInput<TensorsPair>();
  auto rstd = stackGetter.getNextInput<TensorsPair>();
  auto weight_opt = stackGetter.getNextInput<at::optional<TensorsPair>>();

  auto metas = GroupNormBwdMeta(stack);
  auto params = FillNativeGroupNormBwdParams(stack);

  const auto rank = input.pt_t.dim();
  auto layout = [rank]() {
    if (rank == 3) {
      return synapse_helpers::layouts::SynapseLayoutFormat::WCN;
    } else if (rank == 4) {
      return synapse_helpers::layouts::SynapseLayoutFormat::WHCN;
    } else {
      return synapse_helpers::layouts::SynapseLayoutFormat::WHDCN;
    }
  }();

  SetSynapseLayouts(
      {layout,
       layout,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE},
      {layout,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
       synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});

  // Dirty workaround for performance issue. Adding identity nodes to force
  // fallback to graph mode. Fallback to graph mode happens before cguid
  // extraction and is based on number of nodes. That's why we add identity
  // nodes here.  Relates to [SW-209096].
  if (GetExecutionMode() == habana_helpers::HabanaFrontendTypes::EAGER) {
    constexpr size_t num_identities{16};
    for (size_t i = 0; i < num_identities; i++)
      IdentityHelper(graph, grad_in.syn_t, metas[0].shape, metas[0].dtype);
  }

  auto outputs = BuildOp(
      graph,
      GetGuid(),
      {grad_in.syn_t,
       input.syn_t,
       mean.syn_t,
       rstd.syn_t,
       weight_opt.has_value() ? weight_opt.value().syn_t : nullptr},
      {{metas[0].shape, metas[0].dtype, 0},
       {metas[1].shape, metas[1].dtype, 1},
       {metas[2].shape, metas[2].dtype, 2}},
      params.ptr(),
      params.size());

  syn_out(0) = std::move(outputs[0]);
  syn_out(1) = std::move(outputs[1]);
  syn_out(2) = std::move(outputs[2]);
}

} // namespace habana
