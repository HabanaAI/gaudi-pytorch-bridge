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
#include "generated/backend/calculate_scale_for_cast.h"
#include "hpu_ops/custom_op_outshape.h"

namespace habana {

FillParamsT FillCalculateScaleForCastParams(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  const auto reduceAxis = stack[3].toInt();
  const auto fullscale = stack[5].toDouble();
  const auto backoff = stack[6].toDouble();

  const auto ndims = input.dim();
  auto axis = c10::maybe_wrap_dim(reduceAxis, ndims);

  PARAMS_STUB(ns_CalculateScaleForCast::Params);
  params->maxMode =
      static_cast<ns_CalculateScaleForCast::CalculateScaleForCastMaxMode_t>(
          stack[1].toInt());
  params->reduceAxis = ndims - axis - 1;
  params->reduceKeepdim = stack[4].toBool();
  params->maxAbsInputScale = 1.0 / (fullscale * backoff);
  params->scaleMode =
      static_cast<ns_CalculateScaleForCast::CalculateScaleForCastScaleMode_t>(
          stack[2].toInt());

  return paramsT;
}

template <class DimT>
std::vector<DimT> getCalculateScaleForCastOutShape(
    c10::ArrayRef<DimT> inShape,
    ns_CalculateScaleForCast::_CalculateScaleForCastMaxMode_t maxMode,
    int reduceAxis,
    bool reduceKeepdim) {
  std::vector<DimT> outputShape = inShape.vec();

  switch (maxMode) {
    case ns_CalculateScaleForCast::CALCULATE_SCALE_FOR_CAST_NO_MAX:
      break;

    case ns_CalculateScaleForCast::CALCULATE_SCALE_FOR_CAST_MAX_ABS_PTS:
      outputShape = {1};
      break;

    case ns_CalculateScaleForCast::CALCULATE_SCALE_FOR_CAST_MAX_ABS_PCS: {
      const auto ndims = inShape.size();
      auto axis = c10::maybe_wrap_dim(reduceAxis, ndims);

      if (reduceKeepdim)
        outputShape[axis] = 1;
      else
        outputShape.erase(outputShape.begin() + axis);
    }
  }

  return outputShape;
}

OutputMetaDataVector CalculateScaleForCastMeta(const at::Stack& stack) {
  const auto& input = stack_tensor(stack, 0);
  auto maxMode =
      static_cast<ns_CalculateScaleForCast::_CalculateScaleForCastMaxMode_t>(
          stack[1].toInt());
  auto reduceAxis = stack[3].toInt();
  auto reduceKeepdim = stack[4].toBool();

  OutputMetaData meta;
  meta.shape = getCalculateScaleForCastOutShape(
      input.sizes(), maxMode, reduceAxis, reduceKeepdim);
  meta.dtype = input.scalar_type();

  return {meta};
}

sym_sizes_vec calculate_scale_for_cast_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  TORCH_CHECK(inputs.size() == 1, "calculate_scale_for_cast expects 1 input");
  TORCH_CHECK(
      params.size() == 3,
      "calculate_scale_for_cast expects 3 parameters: maxMode, reduceAxis, reduceKeepdim");
  return {getCalculateScaleForCastOutShape(
      inputs[0].sym_sizes(),
      static_cast<ns_CalculateScaleForCast::_CalculateScaleForCastMaxMode_t>(
          params[0]),
      static_cast<int>(params[1]),
      static_cast<bool>(params[2]))};
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(
    calculate_scale_for_cast,
    calculate_scale_for_cast_out_shape);

} // namespace habana
