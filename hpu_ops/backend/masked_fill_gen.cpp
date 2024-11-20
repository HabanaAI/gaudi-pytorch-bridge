/**
* Copyright (c) 2021-2024 Intel Corporation
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
#include "generated/backend/masked_fill.h"

namespace habana {

OutputMetaDataVector MaskedFillMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto mask_shape = stack_tensor(stack, 1).sizes();

  OutputMetaData meta{};
  meta.dtype = self.scalar_type();
  meta.shape = at::infer_size(self.sizes(), mask_shape);

  return {meta};
}

SharedMetaDataVector MaskedFillSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& mask = stack_tensor(stack, 1);
  const auto& value = stack.at(2);
  const auto dtype = self.scalar_type();
  const auto selfRank = self.dim();
  const auto maskRank = mask.dim();
  const auto outputRank = std::max(selfRank, maskRank);

  SharedMetaData maskedFillSharedMeta{"masked_fill_fwd"};
  maskedFillSharedMeta.inputs_data = {
      {selfRank, dtype}, {maskRank, mask.scalar_type()}};
  if (value.isTensor()) {
    // CGUID accepts any type and casts it to self's type
    const auto& valueTensor = value.toTensor();
    maskedFillSharedMeta.inputs_data.emplace_back(valueTensor.dim(), dtype);
  }
  maskedFillSharedMeta.outputs_data.emplace_back(outputRank, dtype);

  return {maskedFillSharedMeta};
}

bool MaskedFillSTMeta(
    habana_helpers::IShapeList& inputs,
    habana_helpers::IShapeList& outputs) {
  static_cast<void>(outputs);
  static_cast<void>(inputs);
  return true;
}

std::shared_ptr<void> FillMaskedFillParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_MaskedFill::ParamsV2);
  auto value = stack.at(2);
  if (value.isTensor()) {
    return params;
  }

  auto self = stack_tensor(stack, 0);
  auto self_dtype = habana_helpers::getInternalDtype(self.scalar_type());

  if ((self_dtype == c10::ScalarType::Long ||
       self_dtype == c10::ScalarType::UInt64) &&
      common::IsInt64Supported()) {
    int64_t val = value.to<int64_t>();
    params->value_low = val;
    params->value_high = val >> 32;
  } else if (c10::isIntegralType(self_dtype, true)) {
    params->value.i = value.toScalar().toInt();
  } else {
    params->value.f = value.toScalar().toFloat();
  }
  return params;
}

void MaskedFill::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  size_t size = 0;
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};

  auto metadata = MaskedFillMeta(stack)[0];
  auto out_shape = metadata.shape;
  auto out_dtype = metadata.dtype;
  const auto& params = FillMaskedFillParams(stack, size);

  auto value = stack.at(2);
  if (value.isTensor()) {
    inputs.push_back(syn_in(2));
  }
  bool check_long = (out_dtype == c10::ScalarType::Long ||
                     out_dtype == c10::ScalarType::UInt64) &&
      common::IsInt64Supported();
  auto guid =
      get_guid_with_precision("masked_fill_fwd", ScalarType(), check_long);
  auto result = BuildOp(
      graph,
      guid,
      std::move(inputs),
      {{out_shape, out_dtype, 0}},
      params.get(),
      size);
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
