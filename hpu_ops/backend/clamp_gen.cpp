/*******************************************************************************
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

#include "generated/backend/_foreach_clamp_max.h"
#include "generated/backend/_foreach_clamp_min.h"
#include "generated/backend/clamp.h"
#include "generated/backend/clamp_max.h"
#include "generated/backend/clamp_min.h"
#include "habana_helpers/dtype_helpers.h"
#include "hpu_ops/backend/foreach.h"
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
static std::shared_ptr<void> FillClampParamsAndSetMinMax(
    const at::Stack& stack,
    size_t& size) {
  ScalarType min = stack[1].isScalar()
      ? stack[1].toScalar().to<ScalarType>()
      : -std::numeric_limits<ScalarType>::max();
  ScalarType max = stack[2].isScalar() ? stack[2].toScalar().to<ScalarType>()
                                       : std::numeric_limits<ScalarType>::max();
  return ClampParams(min, max, size);
}

std::shared_ptr<void> FillClampParams(const at::Stack& stack, size_t& size) {
  auto result_type = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  if (c10::isFloatingType(result_type)) {
    return FillClampParamsAndSetMinMax<float>(stack, size);
  } else {
    return FillClampParamsAndSetMinMax<int>(stack, size);
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

SharedMetaDataVector ClampSharedMeta(const at::Stack& stack) {
  auto dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  auto self = stack_tensor(stack, 0);
  auto selfRank = self.dim();

  auto min = stack.at(1);
  auto max = stack.at(2);
  auto isMinTensor = min.isTensor();
  auto isMaxTensor = max.isTensor();

  SharedMetaData clampSharedMeta{"clamp_pt_fwd"};
  clampSharedMeta.inputs_data.emplace_back(selfRank, dtype);
  clampSharedMeta.inputs_data.push_back(
      isMinTensor ? SharedMetaTensor{min.toTensor().dim(), dtype}
                  : createOptionalNotPresentSharedMetaTensor());
  clampSharedMeta.inputs_data.push_back(
      isMaxTensor ? SharedMetaTensor{max.toTensor().dim(), dtype}
                  : createOptionalNotPresentSharedMetaTensor());

  clampSharedMeta.outputs_data.emplace_back(selfRank, dtype);
  return {clampSharedMeta};
}

SharedMetaDataVector ClampMinSharedMeta(const at::Stack& stack) {
  at::Scalar dummyScalar;
  return ClampSharedMeta({stack.at(0), stack.at(1), dummyScalar});
}

SharedMetaDataVector ClampMaxSharedMeta(const at::Stack& stack) {
  at::Scalar dummyScalar;
  return ClampSharedMeta({stack.at(0), dummyScalar, stack.at(1)});
}

SharedMetaDataVector ForeachClampMinSharedMeta(const at::Stack& stack) {
  SharedMetaCreateFunction sharedMetaCreator = [](const at::Stack& stack) {
    return ClampMinSharedMeta(stack);
  };

  return CommonForeachBinarySharedMeta(stack, sharedMetaCreator);
}

SharedMetaDataVector ForeachClampMaxSharedMeta(const at::Stack& stack) {
  SharedMetaCreateFunction sharedMetaCreator = [](const at::Stack& stack) {
    return ClampMaxSharedMeta(stack);
  };

  return CommonForeachBinarySharedMeta(stack, sharedMetaCreator);
}

static synapse_helpers::tensor ClampCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> inputs,
    at::ScalarType dtype,
    std::vector<int64_t> shape,
    int out_index) {
  return std::move(OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("clamp_pt_fwd", dtype),
       inputs,
       {{shape, dtype, out_index}}})[0]);
}

void clamp::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  StackGetter stackGetter(stack, "clamp::AddNode");
  auto input = getNextInput<TensorsPair>(stackGetter);
  std::vector<synTensor> inputs = {input.syn_t};
  size_t size = 0;
  auto params = FillParams(stack, size);
  const auto compute_type =
      c10::isIntegralType(meta.dtype, true) ? c10::ScalarType::Int : meta.dtype;
  syn_out(0) = std::move(OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("clamp_pt_fwd", compute_type),
       inputs,
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size})[0]);
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

  syn_out(0) = ClampCommon(this, graph, inputs, meta.dtype, meta.shape, 0);
}

void clampMaxTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), nullptr, syn_in(1)};

  syn_out(0) = ClampCommon(this, graph, inputs, meta.dtype, meta.shape, 0);
}

void clampMinTensor::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto meta = OutputMeta(stack)[0];
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};

  syn_out(0) = ClampCommon(this, graph, inputs, meta.dtype, meta.shape, 0);
}

static synapse_helpers::tensor createForeachClampNode(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const std::vector<synTensor>& syn_inputs,
    const std::vector<at::IValue>& pt_inputs,
    int out_index,
    bool is_max) {
  const at::Tensor& self = pt_inputs[0].toTensor();
  if (pt_inputs[1].isTensor()) {
    const at::Tensor& other = pt_inputs[1].toTensor();
    auto result_type = at::result_type(self, other);
    std::vector<synTensor> inputs = syn_inputs;

    if (is_max) {
      inputs = {syn_inputs[0], nullptr, syn_inputs[1]};
    }
    auto outshape = at::infer_size(self.sizes(), other.sizes());

    return ClampCommon(op, graph, inputs, result_type, outshape, out_index);
  } else {
    const at::Scalar& other = pt_inputs[1].toScalar();
    auto result_type = at::result_type(self, other);

    auto syn_other = OpBackend::BuildConstant(op, graph, other, result_type);
    std::vector<synTensor> inputs = {syn_inputs[0], syn_other.get()};
    if (is_max) {
      inputs = {syn_inputs[0], nullptr, syn_other.get()};
    }

    return ClampCommon(
        op, graph, inputs, result_type, self.sizes().vec(), out_index);
  }
}

void ForeachClamp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const size_t size = computeInputsNumber(stack);
  std::vector<synTensor> inputs(size);
  for (size_t i = 0; i < size; i++) {
    inputs[i] = syn_in(i);
  }
  const bool is_max = guid_.find("foreach_clamp_max") != std::string::npos;
  NodeCreateFunction node_creator =
      [is_max](
          OpBackend* op,
          synapse_helpers::graph& graph,
          std::string&,
          const std::vector<synTensor>& syn_inputs,
          const std::vector<at::IValue>& pt_inputs,
          int out_index) {
        return createForeachClampNode(
            op, graph, syn_inputs, pt_inputs, out_index, is_max);
      };

  auto results =
      CommonForeachBinary(this, guid_, inputs, graph, stack, node_creator);
  for (size_t i = 0; i < results.size(); i++) {
    syn_out(i) = std::move(results[i]);
  }
}
} // namespace habana
