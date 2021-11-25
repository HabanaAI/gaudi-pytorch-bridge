/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/hpu_op.h"

#define CHECK_DIM(input_size)                                     \
  TORCH_CHECK(                                                    \
      input_size == 4,                                            \
      "Upsample2D expects input_size equals to 4, but got size ", \
      input_size);

#define CHECK_NULL_INPUT(out_size, scale)                     \
  TORCH_CHECK(                                                \
      !(out_size == c10::nullopt && scale == c10::nullopt) || \
          (out_size != c10::nullopt &&                        \
           (scale != c10::nullopt && !scale.isScalar())),     \
      "Must specify exactly one of output_size and scale_factors for Upsample2D");

#define CHECK_OUTSIZE_OR_SCALE(input)                                       \
  TORCH_CHECK(                                                              \
      input.size() == 2,                                                    \
      "Upsample2D expects output_size or scale size equals to 2, but got ", \
      input.size());

#define CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(                                     \
    input_height, output_height, input_width, output_width)                  \
  TORCH_CHECK(                                                               \
      (input_width > 0 && output_width > 0) &&                               \
          (input_height > 0 && output_height > 0),                           \
      "Input and output sizes should be greater than 0, but got input (W: ", \
      input_width,                                                           \
      ") and (H: ",                                                          \
      input_height,                                                          \
      ") output (W: ",                                                       \
      output_width,                                                          \
      ") for Upsample2D");

namespace habana {

// Forward Output Shape
sizes_vec Upsample2DFwdOutputShape(const at::Stack& stack, bool) {
  bool isBilinear = stack.at(2).isBool();
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = isBilinear ? stack.at(3) : stack.at(2);
  double scale_w = 1, scale_h = 1;
  if (!scale.isNone() && scale.isScalar()) {
    scale_w = isBilinear ? stack.at(4).toDouble() : stack.at(3).toDouble();
    scale_h = isBilinear ? stack.at(3).toDouble() : stack.at(2).toDouble();
  } else if (!scale.isNone()) {
    scale_w = scale.toDoubleVector().at(1);
    scale_h = scale.toDoubleVector().at(0);
  }
  std::vector<int64_t> out_shape;

  CHECK_DIM(self.dim());
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) { // output_sizes must have a value
    CHECK_OUTSIZE_OR_SCALE(out_size.toIntVector());
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        out_size.toIntVector().at(0),
        out_size.toIntVector().at(1)};
  } else if (!scale.isNone()) { // scale factors should have a value
    // if scale is double then scale_h, scale_w are given as double
    if (!scale.isScalar()) {
      CHECK_OUTSIZE_OR_SCALE(scale.toDoubleVector());
    }
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(self.sizes()[2] * scale_h),
        static_cast<int64_t>(self.sizes()[3] * scale_w)};
  }
  CHECK_INPUT_OUTPUT_HEIGHT_WIDTH(
      self.sizes()[2], out_shape.at(2), self.sizes()[3], out_shape.at(3));
  return {out_shape};
}

// Backward Output Shape
sizes_vec Upsample2DBwdOutputShape(const at::Stack& stack, bool) {
  bool isBilinear = stack.at(3).isBool();
  auto grad_out = stack.at(0).toTensor();
  auto input_size = stack.at(2).toIntVector();
  auto out_size = stack.at(1);
  auto scale = isBilinear ? stack.at(4) : stack.at(3);
  CHECK_DIM(grad_out.dim());
  CHECK_DIM(input_size.size());
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) {
    CHECK_OUTSIZE_OR_SCALE(out_size.toIntVector());
  }
  if (!scale.isNone() && !scale.isScalar()) {
    CHECK_OUTSIZE_OR_SCALE(scale.toDoubleVector());
  }
  return {input_size};
}

// AddNode for UpsampleBilinear2D
void UpSampleBilinear2DOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  // Based on align_corner's position, differentiate fwd and bwd ops
  bool isForwardOp = stack.at(2).isBool();

  const torch::Tensor& self = stack.at(0).toTensor();
  auto input_shape = self.sizes();
  auto out_size = stack.at(1);
  auto align_corners =
      isForwardOp ? stack.at(2).toBool() : stack.at(3).toBool();

  // outshape
  auto output_shape = isForwardOp ? Upsample2DFwdOutputShape(stack)[0]
                                  : Upsample2DBwdOutputShape(stack)[0];

  // Transpose - N,C,H,W to N,H,W,C
  std::vector<int64_t> out_shape_4d = {
      input_shape[0], input_shape[2], input_shape[3], input_shape[1]};

  // scales
  auto scales = isForwardOp ? stack.at(3) : stack.at(4);
  double scale_w = 1, scale_h = 1;
  if (!scales.isNone() && scales.isScalar()) {
    scale_h = isForwardOp ? stack.at(3).toDouble() : stack.at(4).toDouble();
    scale_w = isForwardOp ? stack.at(4).toDouble() : stack.at(5).toDouble();
  } else if (!scales.isNone()) {
    scale_h = scales.toDoubleVector().at(0);
    scale_w = scales.toDoubleVector().at(1);
  }

  synTransposeParams trans_params{};
  trans_params.tensorDim = self.dim();
  for (int i = 0; i < self.dim(); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  auto transpose_nhwc = BuildOp(
      graph,
      "transpose",
      {syn_in(0)},
      {{out_shape_4d, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  // Resize kernel
  // modify input height/width values with output height/width values for
  // forward op when both size and scale is provided with
  // align_corners=false
  if (isForwardOp && !align_corners &&
      (!out_size.isNone() && !scales.isNone())) {
    out_shape_4d.at(1) = static_cast<int64_t>(input_shape[2] * scale_h);
    out_shape_4d.at(2) = static_cast<int64_t>(input_shape[3] * scale_w);
  } else {
    out_shape_4d.at(1) = output_shape.at(2);
    out_shape_4d.at(2) = output_shape.at(3);
  }

  size_t size = 0;
  PARAMS_STUB(ns_ResizeKernel::Params);
  params->mode = ResizeInterpolationMode_t::RESIZE_INTER_LINEAR;
  params->nearestMode = ResizeNearestMode_t::FLOOR;
  params->excludeOutside = false;
  params->coordTransMode = align_corners
      ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
      : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
  if (!out_size.isNone()) {
    params->useScales = false;
    params->size1 = out_size.toIntVector().at(1);
    params->size2 = out_size.toIntVector().at(0);
  }
  if (!scales.isNone()) {
    if (!(!out_size.isNone() && align_corners)) {
      params->useScales = true;
      params->scaleDim1 = scale_w;
      params->scaleDim2 = scale_h;
      params->scaleDim3 = 1.0;
    }
  }
  // Use guid_ here for resize_fwd/resize_bwd kernel
  auto resize = BuildOp(
      graph,
      guid_,
      {transpose_nhwc[0].get()},
      {{out_shape_4d, ScalarType()}},
      params.get(),
      size);

  // Slice resize fwd op result when both size and scale is provided
  // with align_corners=false
  if (isForwardOp && !align_corners &&
      (!out_size.isNone() && !scales.isNone())) {
    std::vector<int64_t> slice_shape = {
        input_shape[0], output_shape.at(2), output_shape.at(3), input_shape[1]};

    synSliceParams slice_params{};
    slice_params.ends[0] = input_shape[1]; // C
    slice_params.ends[1] = output_shape.at(3); // W
    slice_params.ends[2] = output_shape.at(2); // H
    slice_params.ends[3] = input_shape[0]; // N

    for (int i = self.dim(); i >= 0; --i) {
      slice_params.axes[i] = i;
      slice_params.starts[i] = 0;
      slice_params.steps[i] = 1;
    }

    resize = BuildOp(
        graph,
        "slice",
        {resize[0].get()},
        {{slice_shape, ScalarType()}},
        &slice_params,
        sizeof(slice_params));
  }

  // Transpose N,H,W,C to N,C,H,W
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  auto transpose_nchw = BuildOp(
      graph,
      "transpose",
      {resize[0].get()},
      {{output_shape, ScalarType(), is_output_persistent_list[0], true}},
      &trans_params,
      sizeof(trans_params));

  syn_out(0) = std::move(transpose_nchw.at(0));
}

// AddNode for UpsampleNearest2D
void UpSampleNearest2DOperator::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const torch::Tensor& self = stack.at(0).toTensor();
  auto input_shape = self.sizes();
  auto out_size = stack.at(1);

  // Based on stack size and scale_h index, differentiate fwd and bwd ops
  bool isForwardOp = (stack.size() == 3) /*vec*/
      || (stack.at(2).isNone() || stack.at(2).isDouble()) /*usual and out*/;

  auto output_shape = isForwardOp ? Upsample2DFwdOutputShape(stack)[0]
                                  : Upsample2DBwdOutputShape(stack)[0];

  // Transpose - N,C,H,W to N,H,W,C
  std::vector<int64_t> out_shape_4d = {
      input_shape[0], input_shape[2], input_shape[3], input_shape[1]};

  // scales
  auto scales = isForwardOp ? stack.at(2) : stack.at(3);
  double scale_w = 1, scale_h = 1;
  if (!scales.isNone() && scales.isScalar()) {
    scale_h = isForwardOp ? stack.at(2).toDouble() : stack.at(3).toDouble();
    scale_w = isForwardOp ? stack.at(3).toDouble() : stack.at(4).toDouble();
  } else if (!scales.isNone()) {
    scale_h = scales.toDoubleVector().at(0);
    scale_w = scales.toDoubleVector().at(1);
  }

  synTransposeParams trans_params{};
  trans_params.tensorDim = self.dim();
  for (int i = 0; i < self.dim(); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  auto transpose_nhwc = BuildOp(
      graph,
      "transpose",
      {syn_in(0)},
      {{out_shape_4d, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  // Resize kernel
  out_shape_4d.at(1) = output_shape.at(2);
  out_shape_4d.at(2) = output_shape.at(3);

  size_t size = 0;
  PARAMS_STUB(ns_ResizeKernel::Params);
  params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
  params->nearestMode = ResizeNearestMode_t::FLOOR;
  params->excludeOutside = false;
  params->coordTransMode =
      ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
  if (!out_size.isNone()) {
    params->useScales = false;
    params->size1 = out_size.toIntVector().at(1);
    params->size2 = out_size.toIntVector().at(0);
  } else if (!scales.isNone()) {
    params->useScales = true;
    params->scaleDim1 = scale_w;
    params->scaleDim2 = scale_h;
    params->scaleDim3 = 1.0;
  }

  // Use guid_ here for resize_fwd/resize_bwd kernel
  auto resize = BuildOp(
      graph,
      guid_,
      {transpose_nhwc[0].get()},
      {{out_shape_4d, ScalarType()}},
      params.get(),
      size);

  // Transpose N,H,W,C to N,C,H,W
  std::swap(trans_params.permutation[1], trans_params.permutation[2]);
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);
  auto transpose_nchw = BuildOp(
      graph,
      "transpose",
      {resize[0].get()},
      {{output_shape, ScalarType(), is_output_persistent_list[0], true}},
      &trans_params,
      sizeof(trans_params));

  syn_out(0) = std::move(transpose_nchw.at(0));
}
} // namespace habana