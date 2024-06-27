/*******************************************************************************
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

  auto weightOpt = stack.at(1).toOptional<at::Tensor>();
  auto biasOpt = stack.at(2).toOptional<at::Tensor>();

  SharedMetaData nativeGroupNormSharedMeta{"native_group_norm_fwd"};
  nativeGroupNormSharedMeta.inputs_data.emplace_back(rank, dtype);
  if (weightOpt.has_value())
    nativeGroupNormSharedMeta.inputs_data.emplace_back(
        weightOpt.value().dim(), dtype);
  else
    nativeGroupNormSharedMeta.inputs_data.push_back(
        createOptionalNotPresentSharedMetaTensor());

  if (biasOpt.has_value())
    nativeGroupNormSharedMeta.inputs_data.emplace_back(
        biasOpt.value().dim(), dtype);
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
  const auto weightOpt = stack.at(4).toOptional<at::Tensor>();
  const auto N = stack.at(5).toInt();
  const auto C = stack.at(6).toInt();
  const auto HxW = stack.at(7).toInt();
  const auto numGroups = stack.at(8).toInt();
  const auto Nmod = N * numGroups;
  const auto inputRank = input.dim();
  const auto inputDtype = input.scalar_type();

  if (N * C * HxW == 0) {
    SharedMetaData memsetSharedMeta{"memset"};
    memsetSharedMeta.outputs_data.emplace_back(inputRank, inputDtype);
    SharedMetaData memsetMeanRstdSharedMeta{"memset"};
    memsetMeanRstdSharedMeta.outputs_data.emplace_back(1, inputDtype);

    return {memsetSharedMeta, memsetMeanRstdSharedMeta};
  }

  SharedMetaDataVector metaVec;
  metaVec.reserve(5);
  SharedMetaTensor bnCommonTensor = {inputRank, inputDtype};
  const bool use_bn_fwd_in_gn_bwd =
      GET_ENV_FLAG_NEW(PT_HPU_USE_BN_FWD_IN_GN_BWD);
  if (use_bn_fwd_in_gn_bwd) {
    SharedMetaData batchNormFwdSharedMeta{"batch_norm_fwd"};
    batchNormFwdSharedMeta.inputs_data = {
        {inputRank, inputDtype},
        {1, c10::ScalarType::Float},
        {1, c10::ScalarType::Float},
        {1, c10::ScalarType::Float},
        {1, c10::ScalarType::Float}};
    batchNormFwdSharedMeta.outputs_data = batchNormFwdSharedMeta.inputs_data;
    metaVec.push_back(batchNormFwdSharedMeta);
  } else {
    SharedMetaData subSharedMeta{"sub_fwd"};
    subSharedMeta.inputs_data = {bnCommonTensor, bnCommonTensor};
    subSharedMeta.outputs_data = {bnCommonTensor};
    metaVec.push_back(subSharedMeta);
  }

  if (!weightOpt.has_value()) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data = {bnCommonTensor};
    metaVec.push_back(constantSharedMeta);
  }

  SharedMetaData multSharedMeta{"mult_fwd"};
  multSharedMeta.inputs_data = {bnCommonTensor, bnCommonTensor};
  multSharedMeta.outputs_data = {bnCommonTensor};
  metaVec.push_back(multSharedMeta);

  if (Nmod > 1) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(1, c10::ScalarType::Float);
    metaVec.push_back(constantSharedMeta);
  }

  SharedMetaData batchNormBwdSharedMeta{"batch_norm_bwd"};
  batchNormBwdSharedMeta.inputs_data = {
      bnCommonTensor,
      bnCommonTensor,
      {1, c10::ScalarType::Float},
      {1, c10::ScalarType::Float},
      {1, c10::ScalarType::Float}};
  batchNormBwdSharedMeta.outputs_data = {
      {bnCommonTensor,
       {1, c10::ScalarType::Float},
       {1, c10::ScalarType::Float}}};
  metaVec.push_back(batchNormBwdSharedMeta);

  SharedMetaData reduceSumMultiDimBetaSharedMeta{"reduce_sum_multi_dim_fwd"};
  reduceSumMultiDimBetaSharedMeta.inputs_data.emplace_back(bnCommonTensor);
  reduceSumMultiDimBetaSharedMeta.outputs_data.emplace_back(1, inputDtype);
  metaVec.push_back(reduceSumMultiDimBetaSharedMeta);

  SharedMetaData reduceSumMultiDimGammaSharedMeta{"reduce_sum_multi_dim_fwd"};
  reduceSumMultiDimGammaSharedMeta.inputs_data.emplace_back(
      gradOut.dim(), inputDtype);
  reduceSumMultiDimGammaSharedMeta.outputs_data.emplace_back(1, inputDtype);
  metaVec.push_back(reduceSumMultiDimGammaSharedMeta);

  return metaVec;
}

std::shared_ptr<void> FillNativeGroupNormParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_NativeGroupNorm::Params);
  params->N = stack[3].toInt();
  params->G = stack[6].toInt();
  params->epsilon = stack[7].toDouble();

  return params;
}

std::shared_ptr<void> FillNativeGroupNormBwdParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_NativeGroupNorm::Params);
  params->N = stack[5].toInt();
  params->G = stack[8].toInt();

  return params;
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
  auto weight = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto bias = stackGetter.getNextInput<c10::optional<TensorsPair>>();
  auto metas = OutputMeta(stack);
  size_t size = 0;
  auto params = FillParams(stack, size);
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
        params.get(),
        size);
    for (unsigned i = 0; i < outputsNumber; i++) {
      syn_out(i) = std::move(outputs[i]);
    }
  }
}

} // namespace habana
