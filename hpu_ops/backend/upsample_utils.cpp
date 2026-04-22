/**
 * Copyright (c) 2025-2026 Intel Corporation
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

#include "upsample_utils.h"
#include "hpu_ops/hpu_op_helper.h"

namespace habana::upsample_utils {

namespace {
constexpr double cubic_coeff =
    -0.75; // As mentioned in TPC guide, value of cubicCoeffA used for
           // cubic interpolation is -0.75.
std::vector<synapse_helpers::tensor> Slice(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType& dtype,
    std::optional<int> final_index = at::nullopt) {
  auto output_size = outshape.size();

  synSliceParamsV2 slice_params{};
  for (int64_t i = output_size - 1; i >= 0; --i) {
    slice_params.axes[i] = i;
    slice_params.starts[i] = 0;
    slice_params.ends[i] = outshape[(output_size - i - 1)];
    slice_params.steps[i] = 1;
  }

  return OpBackend::BuildNode(
      op,
      graph,
      {"slice",
       std::move(input),
       {{outshape, dtype, final_index}},
       &slice_params,
       sizeof(slice_params)});
}

synapse_helpers::tensor UpsampleCommonFuncSynapseLayout(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor>&& input,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    const std::array<double, 3>& scale_dhw,
    const OutputMetaData& meta,
    const at::Tensor self_tensor) {
  auto shape_in_dim = self_tensor.dim();
  auto shape_in = self_tensor.sizes();
  const std::vector<int64_t>* p_shape_out_resize = &meta.shape;

  std::optional<synapse_helpers::tensor> cast_storage;
  auto intermediateDtype = meta.dtype;
  if (meta.dtype == c10::ScalarType::Byte) {
    // u8 to f32
    cast_storage = OpBackend::BuildCast(
        op,
        graph,
        input[0],
        shape_in,
        c10::ScalarType::Byte,
        c10::ScalarType::Float);
    input[0] = cast_storage->get();
    intermediateDtype = c10::ScalarType::Float;
  }
  // Resize
  // modify input width value with output width value
  // when both size and scale is provided with align_corners=false
  bool modifyInputWithOutputWidth =
      isForward && !align_corners && (!out_size.isNone() && !scales.isNone());
  std::vector<int64_t> shape_out_resize;
  if (modifyInputWithOutputWidth) {
    shape_out_resize.reserve(shape_in_dim);
    unsigned scaled_dims = (shape_in_dim > 2) ? shape_in_dim - 2 : 0;

    for (unsigned d = 0; d < shape_in_dim - scaled_dims; ++d) {
      shape_out_resize.push_back(meta.shape[d]);
    }
    for (unsigned d = 0; d < scaled_dims; ++d) {
      shape_out_resize.push_back(
          static_cast<int64_t>(
              shape_in[2 + d] * scale_dhw[d + 3 - scaled_dims]));
    }
    p_shape_out_resize = &shape_out_resize;
  }

  const auto& params = FillResizeParams(
      shape_in_dim,
      upsample_mode,
      out_size,
      scales,
      scale_dhw[2],
      scale_dhw[1],
      scale_dhw[0],
      align_corners,
      false /*antialias*/);
  auto final_index_for_resize =
      modifyInputWithOutputWidth || meta.dtype == c10::ScalarType::Byte
      ? std::optional<int>()
      : std::optional<int>(0);

  auto resize = Resize(
      op,
      graph,
      input,
      *p_shape_out_resize,
      intermediateDtype,
      params,
      final_index_for_resize);
  // Slice
  // For Fwd ops, when both size and scale is provided with align_corners=false
  if (modifyInputWithOutputWidth) {
    auto final_index_for_slice = (meta.dtype == c10::ScalarType::Byte)
        ? std::optional<int>()
        : std::optional<int>(0);

    resize = Slice(
        op,
        graph,
        {resize[0].get()},
        meta.shape,
        intermediateDtype,
        final_index_for_slice);
  };
  if (meta.dtype != c10::ScalarType::Byte) {
    return std::move(resize[0]);
  }
  // f32 to u8
  return OpBackend::BuildCast(
      op, graph, resize[0].get(), meta.shape, intermediateDtype, meta.dtype, 0);
}

} // namespace

// -------> Common checks <-------

void upsample_1d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  HABANA_ASSERT(
      input.dim() == 3,
      "Upsample1D expects input_size equals to 3, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 1,
        "Upsample1D expects out_size equals to 1, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    HABANA_ASSERT(
        scales.toDoubleVector().size() == 1,
        "Upsample1D expects scales equals to 1, but got ",
        scales.toDoubleVector().size());
  }
}

void upsample_2d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  HABANA_ASSERT(
      input.dim() == 4,
      "Upsample2D expects input_size equals to 4, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 2,
        "Upsample2D expects out_size equals to 2, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    HABANA_ASSERT(
        scales.toDoubleVector().size() == 2,
        "Upsample2D expects scales equals to 2, but got ",
        scales.toDoubleVector().size());
  }
}

void upsample_exact_2d_check(const torch::Tensor& input, c10::IValue out_size) {
  HABANA_ASSERT(
      input.dim() == 4,
      "Upsample2D expects input_size equals to 4, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 2,
        "Upsample2D expects out_size equals to 2, but got ",
        out_size.toIntVector().size());
  }
}

void upsample_3d_common_check(
    const torch::Tensor& input,
    c10::IValue out_size,
    c10::IValue scales) {
  HABANA_ASSERT(
      input.dim() == 5,
      "Upsample3D expects input_size equals to 5, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 3,
        "Upsample3D expects out_size equals to 3, but got ",
        out_size.toIntVector().size());
  }

  if (!scales.isNone() && !scales.isScalar()) {
    HABANA_ASSERT(
        scales.toDoubleVector().size() == 3,
        "Upsample3D expects scales equals to 3, but got ",
        scales.toDoubleVector().size());
  }
}

void upsample_exact_3d_check(const torch::Tensor& input, c10::IValue out_size) {
  HABANA_ASSERT(
      input.dim() == 5,
      "Upsample3D expects input_size equals to 5, but got size ",
      input.dim());

  if (!out_size.isNone()) {
    HABANA_ASSERT(
        out_size.toIntVector().size() == 3,
        "Upsample3D expects out_size equals to 3, but got ",
        out_size.toIntVector().size());
  }
}

// -------> Common functions <-------

std::vector<synapse_helpers::tensor> Resize(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input,
    const at::IntArrayRef outshape,
    const at::ScalarType& dtype,
    const FillParamsT& params,
    std::optional<int> final_index) {
  auto guid = op->GetGuid();
  update_guid_dtype(guid, dtype);

  return OpBackend::BuildNode(
      op,
      graph,
      {guid,
       std::move(input),
       {{outshape, dtype, final_index}},
       params.ptr(),
       params.size()});
}

FillParamsT FillResizeParams(
    const int shape_in_dim,
    enum modes upsample_mode,
    c10::IValue out_size,
    c10::IValue scales,
    double scale_w,
    double scale_h,
    double scale_d,
    bool align_corner,
    bool antialias) {
  PARAMS_STUB(ns_ResizeKernel::ParamsAA);
  params->excludeOutside = false;
  params->useAntialiasing = antialias;
  switch (upsample_mode) {
    case nearest:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
      params->nearestMode = ResizeNearestMode_t::FLOOR;
      params->coordTransMode =
          ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
      break;
    case nearest_exact:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
      params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
      params->coordTransMode =
          ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
      break;
    case linear:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_LINEAR;
      params->nearestMode = ResizeNearestMode_t::FLOOR;
      params->coordTransMode = align_corner
          ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
          : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
      break;
    case bicubic:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_CUBIC;
      params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
      params->coordTransMode = align_corner
          ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
          : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
      params->cubicCoeffA =
          cubic_coeff; // As mentioned in TPC guide, value of cubicCoeffA used
                       // for cubic interpolation is -0.75.
      break;
  }
  if (!out_size.isNone()) {
    params->useScales = false;
    if (shape_in_dim == 3) { // 1D variant
      params->size1 = out_size.toIntVector().at(0);
    } else if (shape_in_dim == 4) { // 2D variant
      params->size1 = out_size.toIntVector().at(1);
      params->size2 = out_size.toIntVector().at(0);
    } else if (shape_in_dim == 5) { // 3D variant
      params->size1 = out_size.toIntVector().at(2);
      params->size2 = out_size.toIntVector().at(1);
      params->size3 = out_size.toIntVector().at(0);
    }
    if (align_corner) {
      return paramsT;
    }
  }
  if (!scales.isNone()) {
    params->useScales = true;
    params->scaleDim1 = scale_w;
    params->scaleDim2 = scale_h;
    params->scaleDim3 = scale_d;
  }
  return paramsT;
}

FillParamsT FillResizeParams(
    enum modes upsample_mode,
    std::optional<std::vector<int64_t>> output_size,
    std::optional<std::array<double, 3>> scales,
    bool align_corner,
    bool antialias) {
  PARAMS_STUB(ns_ResizeKernel::ParamsAA);

  params->excludeOutside = false;
  params->useAntialiasing = antialias;

  switch (upsample_mode) {
    case nearest:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
      params->nearestMode = ResizeNearestMode_t::FLOOR;
      params->coordTransMode =
          ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
      break;
    case nearest_exact:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
      params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
      params->coordTransMode =
          ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
      break;
    case linear:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_LINEAR;
      params->nearestMode = ResizeNearestMode_t::FLOOR;
      params->coordTransMode = align_corner
          ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
          : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
      break;
    case bicubic:
      params->mode = ResizeInterpolationMode_t::RESIZE_INTER_CUBIC;
      params->nearestMode = ResizeNearestMode_t::ROUND_DEFAULT;
      params->coordTransMode = align_corner
          ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
          : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
      params->cubicCoeffA = cubic_coeff;
      break;
  }

  if (output_size) {
    const auto spatial_dim = output_size->size();

    params->useScales = false;
    if (spatial_dim == 1) { // 1D variant
      params->size1 = output_size->at(0);
    } else if (spatial_dim == 2) { // 2D variant
      params->size1 = output_size->at(1);
      params->size2 = output_size->at(0);
    } else if (spatial_dim == 3) { // 3D variant
      params->size1 = output_size->at(2);
      params->size2 = output_size->at(1);
      params->size3 = output_size->at(0);
    }
  }

  if (scales) {
    const auto [scale_d, scale_h, scale_w] = *scales;

    params->useScales = true;
    params->scaleDim1 = scale_w;
    params->scaleDim2 = scale_h;
    params->scaleDim3 = scale_d;
  }

  return paramsT;
}

SharedMetaDataVector UpsampleCommmonSharedLayer(
    const at::Stack& stack,
    const bool alignCorners,
    const int64_t scalesIndex,
    const bool isForward,
    const modes upsample_mode,
    const habana_helpers::HabanaExecutionMode execution_mode) {
  const auto& self = stack_tensor(stack, 0);
  const auto& outSize = stack.at(1);
  const auto& scales = stack.at(scalesIndex);
  const bool modifyInputWithOutputWidth =
      isForward && !alignCorners && (!outSize.isNone() && !scales.isNone());

  SharedMetaDataVector metaVec;
  metaVec.reserve(2);

  const auto rank = self.dim();
  auto dtype = self.scalar_type();
  if (dtype == c10::ScalarType::Byte) {
    dtype = c10::ScalarType::Float;
  }
  const std::string guid = isForward ? "resize_fwd" : "resize_bwd";
  SharedMetaTensor commonTensor = {rank, dtype};
  auto& resizeSharedMeta = metaVec.emplace_back(guid);
  resizeSharedMeta.inputs_data = {commonTensor};
  resizeSharedMeta.outputs_data = {commonTensor};
  if (execution_mode == habana_helpers::HabanaExecutionMode::EAGER) {
    const auto& sizes = !isForward ? stack.at(2) : outSize;
    const auto& selfSizes = self.sizes();
    const auto outIndex = !isForward ? rank - 3 : 0;
    const auto dimSelfSize = selfSizes.at(2);
    const auto dimOutSize = sizes.toIntVector().at(outIndex);
    double scale = 0;
    if (!scales.isNone()) {
      scale = !scales.isScalar() ? scales.toDoubleVector().at(0)
                                 : scales.toDouble();
    }
    switch (upsample_mode) {
      case nearest:
      case nearest_exact:
        if (dtype != at::ScalarType::Float && dtype != at::ScalarType::Half &&
            dtype != at::ScalarType::BFloat16 &&
            dtype != at::ScalarType::Char) {
          resizeSharedMeta.options.force_fallback = true;
          resizeSharedMeta.options.fallback_reason =
              "GLUE_INCOMPATIBLE_DATA_TYPE. Resize nearest kernel supports only f32/f16/bf16/i8";
        } else if (!isForward && !scales.isNone() && rank == 5) {
          auto scaledDim =
              std::max(static_cast<int>(std::floor(dimSelfSize / scale)), 1);
          if (scaledDim != dimOutSize) {
            resizeSharedMeta.options.force_fallback = true;
            resizeSharedMeta.options.fallback_reason =
                "GLUE_INCOMPATIBLE_OUTPUT_SIZE. Height dim * (1 / scale) != Output dim (" +
                std::to_string(scaledDim) +
                " != " + std::to_string(dimOutSize) + ").";
          }
        }
        break;
      case linear:
      case bicubic:
        if (dtype != at::ScalarType::Float && dtype != at::ScalarType::Half &&
            dtype != at::ScalarType::BFloat16 && dtype != at::ScalarType::Int &&
            dtype != at::ScalarType::Short && dtype != at::ScalarType::Char) {
          resizeSharedMeta.options.force_fallback = true;
          resizeSharedMeta.options.fallback_reason =
              "GLUE_INCOMPATIBLE_DATA_TYPE. Resize bilinear & bicubic kernel supports only f32/f16/bf16/i32/i16/i8";
        } else if (
            rank == 5 &&
            ((!outSize.isNone() && dimSelfSize != dimOutSize) ||
             (!scales.isNone() && scale != 1.0))) {
          resizeSharedMeta.options.force_fallback = true;
          resizeSharedMeta.options.fallback_reason =
              "GLUE_INCOMPATIBLE_OUTPUT_SIZE. Resize of height dim is only supported in 'nearest' mode.";
        }
        break;
    }
  }

  metaVec.push_back(resizeSharedMeta);
  if (modifyInputWithOutputWidth) {
    auto& sliceSharedMeta = metaVec.emplace_back("slice");
    sliceSharedMeta.inputs_data = {commonTensor};
    sliceSharedMeta.outputs_data = {commonTensor};
  }

  return metaVec;
}

synapse_helpers::tensor UpsampleCommonFunc(
    OpBackend* op,
    synapse_helpers::graph& graph,
    enum modes upsample_mode,
    bool isForward,
    std::vector<synTensor> input,
    c10::IValue out_size,
    bool align_corners,
    c10::IValue scales,
    const std::array<double, 3>& scale_dhw,
    const OutputMetaData& meta,
    const at::Tensor self_tensor) {
  PT_LAZY_DEBUG(__FUNCTION__);
  std::vector<synapse_helpers::tensor> output;
  op->CreateShapeTensorInput(
      graph, meta.dtype, meta.shape, input, SHAPE_TENSOR);
  return UpsampleCommonFuncSynapseLayout(
      op,
      graph,
      upsample_mode,
      isForward,
      std::move(input),
      out_size,
      align_corners,
      scales,
      scale_dhw,
      meta,
      self_tensor);
}

} // namespace habana::upsample_utils
