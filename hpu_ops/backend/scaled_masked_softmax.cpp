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

#include "hpu_ops/scaled_masked_softmax.h"

namespace habana {

std::shared_ptr<void> FillScaledMaskedSoftmaxParams(
    const at::Stack& stack,
    size_t& size) {
  PARAMS_STUB(ns_SmoothL1Kernel::Params);
  params->sigma = stack[2].toDouble();
  return params;
}

ScaledMaskedSoftmax::ScaledMaskedSoftmax(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "scaled_masked_softmax_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetFillParams(FillScaledMaskedSoftmaxParams);
}

OutputMetaDataVector ScaledMaskedTriangularSoftmaxOutputMeta(
    const at::Stack& stack) {
  OutputMetaData meta;
  const auto self = stack[0].toTensor();
  meta.shape = self.sizes().vec();
  meta.dtype =
      stack[6].toOptional<c10::ScalarType>().value_or(self.scalar_type());
  return {meta};
}

ScaledMaskedTriangularSoftmax::ScaledMaskedTriangularSoftmax(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "scaled_masked_triangular_softmax_fwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(ScaledMaskedTriangularSoftmaxOutputMeta);
}

void ScaledMaskedTriangularSoftmax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "ScaledMaskedTriangularSoftmax::AddNode");
  const auto self = stackGetter.getNextInput<TensorsPair>();
  const auto start_end = stackGetter.getNextInput<TensorsPair>();
  const auto inv_scale_attn = stackGetter.getNextInput<double>();
  const auto grouped_batch_size = stackGetter.getNextInput<int>();
  const auto use_max = stackGetter.getNextInput<bool>();
  const auto mode = stackGetter.getNextInput<int>();
  const auto out_dtype =
      stackGetter.getNextInput<c10::optional<c10::ScalarType>>().value_or(
          self.pt_t.scalar_type());
  const auto& input_dtype = ScalarType();

  TORCH_CHECK(
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
  TORCH_CHECK(rank == 3, "Input must be a 3D tensor.");
  TORCH_CHECK(
      shape[1] == shape[2] || shape[1] == 1, "Dim1 must equal (dim2 or 1)");
  TORCH_CHECK(
      grouped_batch_size > 0, "grouped_batch_size must be larger than 0.");
  TORCH_CHECK(
      shape[0] % grouped_batch_size == 0,
      "dim0 must be a multiple of grouped_batch_size.");

  auto start_end_flattend = FlattenHelper(
      graph,
      start_end.syn_t,
      {start_end.pt_t.numel()},
      start_end.pt_t.scalar_type());

  ns_ScaledMaskedSoftmax::Params params{};
  params.invScaleAttn = inv_scale_attn;
  params.groupedBatchSize = grouped_batch_size;
  params.isUseMax = use_max;
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

static const auto& ScaledMaskedSoftmaxKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::scaled_masked_softmax",
            KERNEL_FN_GLOBAL(habana::ScaledMaskedSoftmax))
        .add(
            "hpu::scaled_masked_triangular_softmax",
            KERNEL_FN_GLOBAL(habana::ScaledMaskedTriangularSoftmax));
