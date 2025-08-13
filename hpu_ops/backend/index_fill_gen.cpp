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
#include "generated/backend/index_fill.h"

namespace habana {

FillParamsT FillIndexFillParams(const at::Stack& stack) {
  const auto dim =
      at::maybe_wrap_dim(stack.at(1).toInt(), stack.at(0).toTensor().dim());
  bool is_value_scalar = stack.at(3).isScalar();

  HABANA_ASSERT(
      dim <= std::numeric_limits<int>::max(), "Invalid dimension value: ", dim);
  PARAMS_STUB(ns_IndexFill::Params);
  params->dim = static_cast<int>(dim);
  params->isValueScalar = is_value_scalar;
  if (is_value_scalar) {
    const auto selDtype = stack_tensor(stack, 0).scalar_type();
    const float paramVal = stack.at(3).toScalar().toFloat();
    params->value = selDtype == c10::ScalarType::Bool
        ? static_cast<float>(static_cast<bool>(paramVal))
        : paramVal;
  }

  return paramsT;
}

OutputMetaDataVector IndexFillMeta(const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();

  OutputMetaData meta;
  meta.dtype = input.scalar_type();
  meta.shape = input.sizes().vec();
  return {meta};
}

SharedMetaDataVector IndexFillSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& self = stack_tensor(stack, 0);
  const auto selfRank = self.dim();
  auto computeDtype = self.scalar_type();
  const auto& index = stack_tensor(stack, 2);
  const auto indexRank = index.dim();
  const auto indexDtype = index.scalar_type();
  const auto& value = stack.at(3);

  SharedMetaData indexFillSharedMeta{"index_fill"};
  indexFillSharedMeta.options.allowLongType = true;

  computeDtype = computeDtype == c10::ScalarType::Long ? c10::ScalarType::Int
                                                       : computeDtype;

  indexFillSharedMeta.inputs_data = {
      {selfRank, computeDtype}, {indexRank, indexDtype}};

  if (value.isTensor()) {
    const auto valueTensor = value.toTensor();
    const auto valueRank = valueTensor.dim();
    auto valueDtype = valueTensor.scalar_type();

    valueDtype =
        valueDtype == c10::ScalarType::Long ? c10::ScalarType::Int : valueDtype;

    indexFillSharedMeta.inputs_data.emplace_back(valueRank, valueDtype);
  } else {
    indexFillSharedMeta.inputs_data.push_back(
        createOptionalNotPresentSharedMetaTensor());
  }

  indexFillSharedMeta.outputs_data.emplace_back(selfRank, computeDtype);

  return {indexFillSharedMeta};
}

void IndexFill::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto& input = stack.at(0).toTensor();
  const auto& indexes = stack.at(2).toTensor();
  const bool is_value_scalar = stack.at(3).isScalar();
  const auto meta = IndexFillMeta(stack)[0];

  if (input.sizes().vec().empty()) {
    HABANA_ASSERT(
        indexes.numel() == 1,
        "For input 0-D tensor, number of elements in indices tensor should be 1.");
  }

  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};
  if (!is_value_scalar)
    inputs.push_back(syn_in(2));

  const auto params = FillIndexFillParams(stack);

  auto indexCopyResult = BuildOp(
      graph,
      guid_,
      {std::move(inputs)},
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size());

  syn_out(0) = std::move(indexCopyResult[0]);
}

} // namespace habana
