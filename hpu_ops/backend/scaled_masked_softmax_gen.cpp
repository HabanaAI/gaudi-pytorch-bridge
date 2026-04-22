/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include "generated/backend/scaled_masked_softmax.h"
#include "generated/backend/scaled_masked_triangular_softmax.h"
#include "habana_helpers/conversion.h"
#include "pytorch_helpers/habana_helpers/logging.h"

namespace habana {

FillParamsT FillScaledMaskedSoftmaxParams(const at::Stack& stack) {
  PARAMS_STUB(ns_SmoothL1Kernel::Params);
  params->sigma = static_cast<float>(stack[2].toDouble());
  return paramsT;
}

OutputMetaDataVector ScaledMaskedTriangularSoftmaxMeta(const at::Stack& stack) {
  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  const auto self = stack[0].toTensor();
  meta.shape = self.sizes().vec();
  meta.dtype =
      stack[6].toOptional<c10::ScalarType>().value_or(self.scalar_type());
  return metaVec;
}

SharedMetaDataVector ScaledMaskedTriangularSoftmaxSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const at::Tensor& input = stack_tensor(stack, 0);
  const at::Tensor& startEnd = stack_tensor(stack, 1);
  const at::ScalarType outDtype =
      stack.at(6).toOptional<c10::ScalarType>().value_or(input.scalar_type());

  SharedMetaDataVector meta;
  meta.reserve(2);
  auto& flattenFwdSharedMeta = meta.emplace_back("flatten_fwd");
  flattenFwdSharedMeta.inputs_data = {getSharedMetaFromTensor(startEnd)};
  flattenFwdSharedMeta.outputs_data = {{1, startEnd.scalar_type()}};

  auto& sharedMeta = meta.emplace_back("scaled_masked_triangular_softmax_fwd");
  sharedMeta.inputs_data = {
      getSharedMetaFromTensor(input), {1, startEnd.scalar_type()}};
  sharedMeta.outputs_data.emplace_back(input.dim(), outDtype);

  return meta;
}

void ScaledMaskedTriangularSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "ScaledMaskedTriangularSoftmax::AddNode");
  const auto self = stackGetter.getNextInput<TensorsPair>();
  const auto start_end = stackGetter.getNextInput<TensorsPair>();
  const auto inv_scale_attn = stackGetter.getNextInput<double>();
  const auto grouped_batch_size = stackGetter.getNextInput<long>();
  const auto use_max = stackGetter.getNextInput<bool>();
  const auto mode = stackGetter.getNextInput<long>();
  const auto out_dtype =
      stackGetter.getNextInput<std::optional<c10::ScalarType>>().value_or(
          self.pt_t.scalar_type());
  const auto& input_dtype = ScalarType();

  HABANA_ASSERT(
      input_dtype == out_dtype or
          (input_dtype == c10::ScalarType::BFloat16 and
           (out_dtype == c10::ScalarType::Float8_e5m2 or
            out_dtype == c10::ScalarType::Float8_e4m3fn)),
      "Input and output dtypes must be equal or input must be bfloat16 and output must be fp8. Got: input = ",
      input_dtype,
      ", output = ",
      out_dtype);

  auto shape = self.pt_t.sizes().vec();
  auto rank = shape.size();
  HABANA_ASSERT(rank == 3, "Input must be a 3D tensor.");
  HABANA_ASSERT(
      shape[1] == shape[2] || shape[1] == 1, "Dim1 must equal (dim2 or 1)");
  HABANA_ASSERT(
      grouped_batch_size > 0, "grouped_batch_size must be larger than 0.");
  HABANA_ASSERT(
      shape[0] % grouped_batch_size == 0,
      "dim0 must be a multiple of grouped_batch_size.");

  auto start_end_flattend = FlattenHelper(
      graph,
      start_end.syn_t,
      {start_end.pt_t.numel()},
      start_end.pt_t.scalar_type());

  ns_ScaledMaskedSoftmax::Params params{};
  params.invScaleAttn = static_cast<float>(inv_scale_attn);
  params.groupedBatchSize = safe_convert<unsigned int>(grouped_batch_size);
  params.isUseMax = static_cast<unsigned int>(use_max);
  params.expMode = static_cast<ScaledMaskedSoftmaxExpMode_t>(mode);

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {guid_,
       {self.syn_t, start_end_flattend.get()},
       {{self.pt_t.sizes().vec(), out_dtype, 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

} // namespace habana
