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
#include "generated/backend/cast_to_fp8_just_in_time.h"

namespace habana {

FillParamsT CastToFp8JustInTimeParams(const at::Stack& stack) {
  const auto block_shape = stack[1].toIntVector();
  PARAMS_STUB(ns_ConvertToFp8JustInTime::Params);
  params->block_height = block_shape[0];
  params->block_width = block_shape[1];
  return paramsT;
}

OutputMetaDataVector CastToFp8JustInTimeMeta(const at::Stack& stack) {
  const auto& input_tensor = stack_tensor(stack, 0);
  const auto input_shape = input_tensor.sizes().vec();
  const auto rank = input_shape.size();
  const auto block_shape = stack[1].toIntVector();
  const auto block_height = block_shape[0];
  const auto block_width = block_shape[1];

  TORCH_CHECK(
      rank >= 2,
      "convert_to_fp8_just_in_time: input tensor must have at least 2 "
      "dimensions, got ",
      rank);

  TORCH_CHECK(
      block_height == 1 or block_width == 1 or block_height == block_width,
      "convert_to_fp8_just_in_time: block_height and block_width must be "
      "equal or one of them must be 1, got ",
      block_height,
      " and ",
      block_width);

  TORCH_CHECK(
      input_shape[rank - 2] % block_height == 0 and
          input_shape[rank - 1] % block_width == 0,
      "convert_to_fp8_just_in_time: two last dimensions of input tensor must "
      "be divisible by ",
      "block_height and block_width, got [",
      input_shape[rank - 2],
      ", ",
      input_shape[rank - 1],
      "] and [",
      block_height,
      ", ",
      block_width,
      "]");

  const auto out_dtype = stack[2].isNone() ? at::ScalarType::Float8_e4m3fn
                                           : stack[2].toScalarType();
  const auto scale_dtype =
      stack[3].isNone() ? input_tensor.scalar_type() : stack[3].toScalarType();

  auto scale_shape = input_shape;
  scale_shape[rank - 2] /= block_height;
  scale_shape[rank - 1] /= block_width;

  OutputMetaData meta_output(out_dtype, input_shape);
  OutputMetaData meta_scale(scale_dtype, scale_shape);

  return {meta_output, meta_scale};
}

} // namespace habana
