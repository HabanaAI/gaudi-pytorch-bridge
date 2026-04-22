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
#include <limits>
#include "generated/backend/cast_to_fp8_hybrid.h"

namespace habana {

// Determines if STOCHASTIC_FLUSH_TO_ZERO should be used instead
// of STORCHASTIC_ROUNDING.
static const bool is_sr_sftz =
    GET_ENV_FLAG_NEW(PT_HPU_STOCHASTIC_ROUNDING_MODE) == 1;

FillParamsT CastToFp8HybridParams(const at::Stack& stack) {
  const bool stochastic_rounding = stack[3].toBool();
  PARAMS_STUB(ns_CastKernel::Params);
  if (stochastic_rounding) {
    const bool is_sftz_available = is_sr_sftz and
        stack_tensor(stack, 0).scalar_type() == at::ScalarType::BFloat16;
    params->round_mode = is_sftz_available ? CAST_ROUND_SFTZ : CAST_ROUND_SR;
  } else {
    params->round_mode = CAST_ROUND_HALF_NE;
  }
  return paramsT;
}

OutputMetaDataVector CastToFp8HybridMeta(const at::Stack& stack) {
  auto input_sv = stack_tensor(stack, 0).sizes().vec();
  bool is_amax = stack[4].toBool();
  std::vector<int64_t> amax_shape{};
  if (not is_amax) {
    amax_shape.push_back(0);
  }

  OutputMetaDataVector metaVec(3);

  auto& meta_152 = metaVec[0];
  meta_152.dtype = at::ScalarType::Float8_e5m2;
  meta_152.shape = input_sv;

  auto& meta_143 = metaVec[1];
  meta_143.dtype = at::ScalarType::Float8_e4m3fn;
  meta_143.shape = input_sv;

  auto& meta_amax = metaVec[2];
  meta_amax.dtype = at::ScalarType::Float;
  meta_amax.shape = amax_shape;
  return metaVec;
}

SharedMetaDataVector CastToFp8HybridSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const at::Tensor& input = stack.at(0).toTensor();
  HABANA_ASSERT(
      input.dim() <= std::numeric_limits<int>::max(),
      "Input tensor dimension exceeds maximum allowed value.");
  const auto inputDim = static_cast<int>(input.dim());
  const bool isAmax = stack.at(4).toBool();

  const bool is152Scale = stack.at(1).isTensor();
  const bool is143Scale = stack.at(2).isTensor();

  SharedMetaDataVector meta;
  meta.reserve(1);
  auto& sharedMeta = meta.emplace_back("convert_to_fp8_hybrid");
  sharedMeta.inputs_data.emplace_back(inputDim, input.scalar_type());
  sharedMeta.inputs_data.push_back(
      is152Scale ? SharedMetaTensor{1, c10::ScalarType::Float}
                 : createOptionalNotPresentSharedMetaTensor());
  sharedMeta.inputs_data.push_back(
      is143Scale ? SharedMetaTensor{1, c10::ScalarType::Float}
                 : createOptionalNotPresentSharedMetaTensor());

  sharedMeta.outputs_data.emplace_back(inputDim, c10::ScalarType::Float8_e5m2);
  sharedMeta.outputs_data.emplace_back(
      inputDim, c10::ScalarType::Float8_e4m3fn);

  if (isAmax) {
    sharedMeta.outputs_data.emplace_back(1, c10::ScalarType::Float);
  }

  return meta;
}

void CastToFp8Hybrid::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "CastToFp8Hybrid::AddNode");
  auto self = stackGetter.getNextInput<TensorsPair>();
  auto scale_152 = stackGetter.getNextInput<std::optional<TensorsPair>>();
  auto scale_143 = stackGetter.getNextInput<std::optional<TensorsPair>>();
  stackGetter.getNextInput<bool>();
  auto is_amax = stackGetter.getNextInput<bool>();
  auto src_type = self.pt_t.scalar_type();

  HABANA_ASSERT(
      src_type == at::ScalarType::Float or src_type == at::ScalarType::BFloat16,
      "CastToFp8Hybrid input must be of float or bfloat16 dtype.");

  auto out_meta = CastToFp8HybridMeta(stack);

  std::vector<synTensor> syn_inputs{self.syn_t};
  syn_inputs.push_back(scale_152 ? scale_152->syn_t : nullptr);
  syn_inputs.push_back(scale_143 ? scale_143->syn_t : nullptr);

  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {out_meta[0].shape, out_meta[0].dtype, 0},
      {out_meta[1].shape, out_meta[1].dtype, 1}};
  if (is_amax) {
    output_attrs.push_back({out_meta[2].shape, out_meta[2].dtype, 2});
  }

  const auto& params = CastToFp8HybridParams(stack);

  auto casted = OpBackend::BuildNode(
      this,
      graph,
      {guid_, syn_inputs, output_attrs, params.ptr(), params.size()});

  syn_out(0) = std::move(casted[0]);
  syn_out(1) = std::move(casted[1]);
  if (is_amax) {
    syn_out(2) = std::move(casted[2]);
  }
}

} // namespace habana
