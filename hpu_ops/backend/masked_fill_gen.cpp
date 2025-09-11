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
#include "generated/backend/masked_fill.h"
#include "pytorch_helpers/habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

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
    habana_helpers::HabanaExecutionMode /*unused*/) {
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

FillParamsT FillMaskedFillParams(const at::Stack& stack) {
  PARAMS_STUB(ns_MaskedFill::ParamsV2);
  auto value = stack.at(2);
  if (value.isTensor()) {
    return paramsT;
  }

  auto self = stack_tensor(stack, 0);
  auto self_dtype = habana_helpers::getInternalDtype(self.scalar_type());
  const auto scalarValue = value.toScalar();
  if ((self_dtype == c10::ScalarType::Long ||
       self_dtype == c10::ScalarType::UInt64) &&
      common::IsInt64Supported()) {
    params->value_low = safe_convert<int>(
        scalarValue.isIntegral(true)
            ? scalarValue.to<int64_t>()
            : static_cast<int64_t>(scalarValue.toFloat()));
    params->value_high = safe_convert<int>(
        (scalarValue.isIntegral(true)
             ? scalarValue.to<int64_t>()
             : static_cast<int64_t>(scalarValue.toFloat())) >>
        32);
  } else if (c10::isIntegralType(self_dtype, true)) {
    params->value.i = scalarValue.isIntegral(true)
        ? scalarValue.toInt()
        : static_cast<int32_t>(scalarValue.toFloat());
  } else {
    params->value.f = scalarValue.isIntegral(true)
        ? static_cast<float>(scalarValue.toInt())
        : scalarValue.toFloat();
  }
  return paramsT;
}

void MaskedFill::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1)};

  auto metadata = MaskedFillMeta(stack)[0];
  auto out_shape = metadata.shape;
  auto out_dtype = metadata.dtype;
  const auto& params = FillMaskedFillParams(stack);

  auto value = stack.at(2);
  if (value.isTensor()) {
    inputs.push_back(syn_in(2));
  }
  bool check_long = (out_dtype == c10::ScalarType::Long ||
                     out_dtype == c10::ScalarType::UInt64) &&
      common::IsInt64Supported();
  using namespace std::literals;
  auto guid =
      get_guid_with_precision("masked_fill_fwd"sv, ScalarType(), check_long);
  auto result = BuildOp(
      graph,
      guid,
      std::move(inputs),
      {{out_shape, out_dtype, 0}},
      params.ptr(),
      params.size());
  syn_out(0) = std::move(result[0]);
}

} // namespace habana
