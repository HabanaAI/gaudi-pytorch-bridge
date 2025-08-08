/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/block_softmax_const_max.h"

namespace habana {

namespace {
at::ScalarType getOutDtype(const at::Stack& stack) {
  return stack[6].toOptional<c10::ScalarType>().value_or(
      at::ScalarType::BFloat16);
}
} // namespace

OutputMetaDataVector BlockSoftmaxConstMaxMeta(const at::Stack& stack) {
  const auto& attn = stack_tensor(stack, 0);
  const auto out_dtype = getOutDtype(stack);

  TORCH_CHECK(
      out_dtype == at::ScalarType::BFloat16 ||
          out_dtype == at::ScalarType::Float8_e4m3fn,
      "Supported output dtypes are bfloat16 and float8_e4m3, got ",
      out_dtype);

  OutputMetaData meta;
  meta.shape = attn.sizes().vec();
  meta.dtype = out_dtype;

  return {meta};
}

FillParamsT BlockSoftmaxConstMaxParams(const at::Stack& stack) {
  const auto batch_size = stack.at(3).toInt();
  const auto global_block_max = stack.at(4).toDouble();
  const auto output_scale = stack.at(5).toDouble();
  const auto out_dtype = getOutDtype(stack);
  const auto mode = out_dtype == at::ScalarType::BFloat16
      ? BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_BF16_OUT
      : BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_HF8_OUT;

  PARAMS_STUB(ns_BlockSoftmaxConstantMax::Params);
  params->batchSize = batch_size;
  params->outputScale = output_scale;
  params->globalBlockMax = global_block_max;
  params->mode = mode;
  return paramsT;
}

} // namespace habana
