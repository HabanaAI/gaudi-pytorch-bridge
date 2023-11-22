/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/clamp.h"
#include "generated/backend/clamp_max.h"
#include "generated/backend/clamp_min.h"
#include "habana_helpers/dtype_helpers.h"

namespace habana {

OutputMetaDataVector ClampMeta(const at::Stack& stack) {
  OutputMetaData meta{};
  auto selfSizes = stack_tensor(stack, 0).sizes();
  bool minMaxScalar = stack.at(1).isScalar() || stack.at(2).isScalar();
  bool minTensorDefined = stack.at(1).isTensor();
  bool maxTensorDefined = stack.at(2).isTensor();

  if (minMaxScalar) {
    meta.shape = selfSizes.vec();
  } else {
    if (minTensorDefined && maxTensorDefined)
      meta.shape = at::infer_size(
          at::infer_size(selfSizes, stack_tensor(stack, 1).sizes()),
          stack_tensor(stack, 2).sizes());
    else if (minTensorDefined)
      meta.shape = at::infer_size(selfSizes, stack_tensor(stack, 1).sizes());
    else
      meta.shape = at::infer_size(selfSizes, stack_tensor(stack, 2).sizes());
  }

  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);

  return {meta};
}

template <typename ScalarType>
static std::shared_ptr<void> ClampParams(
    ScalarType min,
    ScalarType max,
    size_t& size) {
  PARAMS_STUB(ns_ClampKernel::Params);

  get<ScalarType>(params->lowerBound) = min;
  get<ScalarType>(params->upperBound) = max;

  return params;
}

template <typename ScalarType>
static std::shared_ptr<void> FillClampParamsAndFixOverlapping(
    const at::Stack& stack,
    size_t& size) {
  ScalarType min = stack[1].isScalar()
      ? stack[1].toScalar().to<ScalarType>()
      : -std::numeric_limits<ScalarType>::max();
  ScalarType max = stack[2].isScalar() ? stack[2].toScalar().to<ScalarType>()
                                       : std::numeric_limits<ScalarType>::max();
  if (max < min) {
    min = max;
  }
  return ClampParams(min, max, size);
}

std::shared_ptr<void> FillClampParams(const at::Stack& stack, size_t& size) {
  auto result_type = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  if (c10::isFloatingType(result_type)) {
    return FillClampParamsAndFixOverlapping<float>(stack, size);
  } else {
    return FillClampParamsAndFixOverlapping<int>(stack, size);
  }
}

std::shared_ptr<void> FillClampMinParams(const at::Stack& stack, size_t& size) {
  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          stack, c10::nullopt, false);

  c10::ScalarType result_type = dtype_helper.get_result_dtype();

  if (c10::isFloatingType(result_type)) {
    return ClampParams(
        stack[1].toScalar().toFloat(), std::numeric_limits<float>::max(), size);
  }
  return ClampParams(
      stack[1].toScalar().toInt(), std::numeric_limits<int>::max(), size);
}

std::shared_ptr<void> FillClampMaxParams(const at::Stack& stack, size_t& size) {
  auto dtype_helper =
      habana_helpers::DTypeHelper::binary_op_with_type_promotion(
          stack, c10::nullopt, false);

  c10::ScalarType result_type = dtype_helper.get_result_dtype();

  if (c10::isFloatingType(result_type)) {
    return ClampParams(
        -std::numeric_limits<float>::max(),
        stack[1].toScalar().toFloat(),
        size);
  }
  return ClampParams(
      -std::numeric_limits<int>::max(), stack[1].toScalar().toInt(), size);
}

static synapse_helpers::tensor ClampCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    const OutputMetaData& meta) {
  return std::move(OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("clamp_pt_fwd", meta.dtype),
       inputs,
       {{meta.shape, meta.dtype, 0}}})[0]);
}

void clampTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  bool minTensorDefined = stack.at(1).isTensor();
  bool maxTensorDefined = stack.at(2).isTensor();
  HABANA_ASSERT(
      maxTensorDefined || minTensorDefined,
      "At least one of 'min' or 'max' must not be None")

  StackGetter stackGetter(stack, "clampTensor::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  auto min = getNextInput<c10::optional<TensorsPair>>(stackGetter);
  auto max = getNextInput<c10::optional<TensorsPair>>(stackGetter);

  std::vector<synTensor> inputs = {input.syn_t};
  inputs.push_back(min ? min.value().syn_t : nullptr);
  inputs.push_back(max ? max.value().syn_t : nullptr);

  syn_out(0) = ClampCommon(this, graph, inputs, meta);
}

void clampMaxTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), nullptr, syn_in(1)};

  syn_out(0) = ClampCommon(this, graph, inputs, meta);
}

void clampMinTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};

  syn_out(0) = ClampCommon(this, graph, inputs, meta);
}

} // namespace habana
