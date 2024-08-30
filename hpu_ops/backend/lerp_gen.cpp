/******************************************************************************
 * Copyright (C) 2021-2024 Habana Labs, Ltd. an Intel Company
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
#include "generated/backend/_foreach_lerp.h"
#include "generated/backend/lerp.h"

namespace habana {

OutputMetaData SingleLerpMeta(
    const at::Tensor& start,
    const at::Tensor& end,
    const at::IValue& weight) {
  std::vector<int64_t> shape = at::infer_size(start.sizes(), end.sizes());
  if (weight.isTensor()) {
    shape = at::infer_size(shape, weight.toTensor().sizes());
  }
  return OutputMetaData{start.scalar_type(), shape};
}

OutputMetaDataVector LerpMeta(const at::Stack& stack) {
  return {SingleLerpMeta(
      stack_tensor(stack, 0), stack_tensor(stack, 1), stack.at(2))};
}

OutputMetaDataVector ForeachLerpMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVector;
  const auto& self = stack.at(0).toTensorList();
  const auto& tensor1 = stack.at(1).toTensorList();

  const size_t size = self.size();
  metaVector.reserve(size);

  for (size_t i = 0; i < size; ++i) {
    const at::IValue& weight =
        stack.at(2).isScalar() ? stack.at(2) : stack.at(2).toList()[i];
    metaVector.emplace_back(SingleLerpMeta(self[i], tensor1[i], weight));
  }

  return metaVector;
}

SharedMetaDataVector LerpSharedMeta(const at::Stack& stack) {
  auto start = stack_tensor(stack, 0);
  auto startRank = start.dim();
  auto dtype = start.scalar_type();
  auto end = stack_tensor(stack, 1);
  auto endRank = end.dim();
  auto weight = stack.at(2);
  int64_t weightRank = 1;
  if (weight.isTensor()) {
    auto weightTensor = weight.toTensor();
    weightRank = weightTensor.dim();
  };

  auto subOutputRank = std::max(startRank, endRank);
  SharedMetaData subSharedMeta{"sub"};
  subSharedMeta.inputs_data = {{endRank, dtype}, {startRank, dtype}};
  subSharedMeta.outputs_data.emplace_back(subOutputRank, dtype);

  auto multOutputRank = std::max(subOutputRank, weightRank);
  SharedMetaData multSharedMeta{"mult"};
  multSharedMeta.inputs_data = {
      subSharedMeta.outputs_data[0], {weightRank, dtype}};
  multSharedMeta.outputs_data.emplace_back(multOutputRank, dtype);

  auto addOutputRank = std::max(startRank, multOutputRank);
  SharedMetaData addSharedMeta{"add"};
  addSharedMeta.inputs_data = {
      {startRank, dtype}, {multSharedMeta.outputs_data[0]}};
  addSharedMeta.outputs_data.emplace_back(addOutputRank, dtype);

  return {subSharedMeta, multSharedMeta, addSharedMeta};
}

SharedMetaDataVector ForeachLerpSharedMeta(const at::Stack& stack) {
  const auto& starts = stack.at(0).toList();
  auto startsSize = starts.size();
  const auto& ends = stack.at(1).toList();
  auto weightTensors = stack.at(2);
  c10::optional<c10::List<c10::IValue>> weightTensorList = c10::nullopt;
  if (weightTensors.isTensorList()) {
    weightTensorList = weightTensors.toList();
  }

  SharedMetaDataVector metaVec;
  metaVec.reserve(startsSize * 3);
  for (size_t i = 0; i < startsSize; i++) {
    c10::IValue weight = weightTensorList.has_value()
        ? weightTensorList.value()[i]
        : c10::IValue();
    at::Stack lerpStack = {starts[i], ends[i], weight};
    auto lerpSharedMeta = LerpSharedMeta(lerpStack);
    metaVec.insert(
        std::end(metaVec),
        std::begin(lerpSharedMeta),
        std::end(lerpSharedMeta));
  }
  return metaVec;
}

synapse_helpers::tensor CommonLerp(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<synTensor>& syn_inputs,
    const std::vector<at::IValue>& pt_inputs,
    const OutputMetaData& meta,
    const int final_result_index,
    const bool isWeightTensor = true) {
  const auto sub_outshape = at::infer_size(
      pt_inputs[0].toTensor().sizes(), pt_inputs[1].toTensor().sizes());
  // subtraction of start and end
  auto sub = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("sub", meta.dtype),
       {syn_inputs[1], syn_inputs[0]},
       {{{sub_outshape}, meta.dtype}}});

  std::optional<synapse_helpers::tensor> constant{};
  if (!isWeightTensor) {
    constant = OpBackend::BuildConstant(
        op, graph, pt_inputs[2].toScalar(), meta.dtype, {1});
  }

  // multiplication of weight and sub
  auto mult = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult", meta.dtype),
       {sub[0].get(), isWeightTensor ? syn_inputs[2] : constant.value().get()},
       {{meta.shape, meta.dtype}}});

  // addition of start and mult
  auto lerp = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("add", meta.dtype),
       {syn_inputs[0], mult[0].get()},
       {{meta.shape, meta.dtype, final_result_index}}});

  return std::move(lerp[0]);
}

void Lerp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  syn_out(0) = CommonLerp(
      this,
      graph,
      {syn_in(0), syn_in(1), syn_in(2)},
      {stack_tensor(stack, 0), stack_tensor(stack, 1)},
      LerpMeta(stack)[0],
      0);
}

void ForeachLerp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = ForeachLerpMeta(stack);
  const auto& self = stack.at(0).toList();
  const auto& tensor1 = stack.at(1).toList();
  const bool isWeightTensorList = stack.at(2).isTensorList();
  const size_t size = self.size();

  for (size_t i = 0; i < size; ++i) {
    const at::IValue& weight =
        isWeightTensorList ? stack.at(2).toList()[i] : stack.at(2);
    std::vector<synTensor> syn_inputs{syn_in(i), syn_in(i + size)};
    if (isWeightTensorList) {
      syn_inputs.push_back(syn_in(i + 2 * size));
    }
    std::vector<at::IValue> pt_inputs{self[i], tensor1[i], weight};

    syn_out(i) = CommonLerp(
        this, graph, syn_inputs, pt_inputs, meta[i], i, isWeightTensorList);
  }
}
} // namespace habana
