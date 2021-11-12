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

#define CHECK_DIM(input_size)                                           \
  TORCH_CHECK(                                                          \
      input_size == 3,                                                  \
      "UpsampleLinear1D expects input_size equals to 3, but got size ", \
      input_size);

#define CHECK_NULL_INPUT(out_size, scale)                     \
  TORCH_CHECK(                                                \
      !(out_size == c10::nullopt && scale == c10::nullopt) || \
          (out_size != c10::nullopt &&                        \
           (scale != c10::nullopt && !scale.isScalar())),     \
      "Must specify exactly either one of output_size or scale_factors for UpsampleLinear1D");

#define CHECK_OUTSIZE_OR_SCALE(vec_input)                                         \
  TORCH_CHECK(                                                                    \
      vec_input.size() == 1,                                                      \
      "UpsampleLinear1D expects output_size or scale size equals to 1, but got ", \
      vec_input.size());

#define CHECK_INPUT_OUTPUT_WIDTH(input_width, output_width)                  \
  TORCH_CHECK(                                                               \
      input_width > 0 && output_width > 0,                                   \
      "Input and output sizes should be greater than 0, but got input (W: ", \
      input_width,                                                           \
      ") and output (W: ",                                                   \
      output_width,                                                          \
      ")");

namespace habana {

// Forward Output Shape
sizes_vec UpsampleLinear1DFwdOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(0).toTensor();
  auto out_size = stack.at(1);
  auto scale = stack.at(3);
  std::vector<int64_t> out_shape;

  CHECK_DIM(self.dim());
  CHECK_NULL_INPUT(out_size, scale);
  if (!out_size.isNone()) {
    CHECK_OUTSIZE_OR_SCALE(out_size.toIntVector());
    out_shape = {
        self.sizes()[0], self.sizes()[1], out_size.toIntVector().at(0)};
  } else if (!scale.isNone()) {
    if (!scale.isScalar()) {
      CHECK_OUTSIZE_OR_SCALE(scale.toDoubleVector());
    }
    double scale_factor =
        scale.isScalar() ? scale.toDouble() : scale.toDoubleVector().at(0);
    auto width = self.sizes()[2];
    out_shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }
  CHECK_INPUT_OUTPUT_WIDTH(self.sizes()[2], out_shape.at(2));
  return {out_shape};
}

// Backward Output Shape
sizes_vec UpsampleLinear1DBwdOutputShape(const at::Stack& stack, bool) {
  auto grad_out = stack.at(0).toTensor();
  auto input_size = stack.at(2).toIntVector();
  auto out_size = stack.at(1);
  auto scale = stack.at(4);
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

std::shared_ptr<void> FillUpsampleLinear1DParams(
    const at::Stack& stack,
    size_t& size) {
  bool isForwardOp = stack.at(2).isBool();
  auto out_size = stack.at(1);
  bool is_align_corner =
      isForwardOp ? stack.at(2).toBool() : stack.at(3).toBool();
  auto scales = isForwardOp ? stack.at(3) : stack.at(4);

  PARAMS_STUB(ns_ResizeKernel::Params);
  params->mode = ResizeInterpolationMode_t::RESIZE_INTER_LINEAR;
  params->nearestMode = ResizeNearestMode_t::FLOOR;
  params->excludeOutside = false;
  params->coordTransMode = is_align_corner
      ? ResizeCoordinateTransformationMode_t::ALIGN_CORNERS_MODE
      : ResizeCoordinateTransformationMode_t::PYTORCH_HALF_PIXEL_MODE;
  if (!out_size.isNone()) {
    params->useScales = false;
    params->size1 = out_size.toIntVector().at(0);
    params->size2 = 1;
    params->size3 = 1;
    if (is_align_corner == true) {
      return params;
    }
  }
  if (!scales.isNone()) {
    params->useScales = true;
    params->scaleDim1 =
        scales.isScalar() ? scales.toDouble() : scales.toDoubleVector().at(0);
    params->scaleDim2 = 1.0;
    params->scaleDim3 = 1.0;
  }
  return params;
}

void UpsampleLinear1DOperator::AddNode(
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
  auto scales = isForwardOp ? stack.at(3) : stack.at(4);
  auto output_shape = isForwardOp ? UpsampleLinear1DFwdOutputShape(stack)[0]
                                  : UpsampleLinear1DBwdOutputShape(stack)[0];

  // Transpose - N,C,W to N,W,C
  synTransposeParams trans_params{};
  trans_params.tensorDim = self.dim();
  for (int i = 0; i < self.dim(); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);
  std::vector<int64_t> out_shape_3d = {
      input_shape[0], input_shape[2], input_shape[1]}; // N,W,C

  auto transpose_nwc = BuildOp(
      graph,
      "transpose",
      {syn_in(0)},
      {{out_shape_3d, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  // Reshape N,W,C to N,H,W,C where H=1
  std::vector<int64_t> out_shape_4d = {
      input_shape[0], static_cast<int64_t>(1), input_shape[2], input_shape[1]};

  auto reshape_nhwc = BuildOp(
      graph,
      "reshape",
      {transpose_nwc[0].get()},
      {{out_shape_4d, ScalarType()}});

  // Resize kernel
  size_t size = 0;
  const auto& params = FillUpsampleLinear1DParams(stack, size);

  // modify input width value with output width value for forward op
  // when both size and scale is provided with align_corners=false
  out_shape_4d.at(2) =
      isForwardOp && !align_corners && (!out_size.isNone() && !scales.isNone())
      ? static_cast<int64_t>(input_shape[2] * scales.toDouble())
      : output_shape.at(2);

  // Use guid_ for resize_fwd/resize_bwd kernel
  auto resize = BuildOp(
      graph,
      guid_,
      {reshape_nhwc[0].get()},
      {{out_shape_4d, ScalarType()}},
      params.get(),
      size);

  // Slice the resize fwd op result when both size and scale is provided with
  // align_corners=false
  if (isForwardOp && !align_corners &&
      (!out_size.isNone() && !scales.isNone())) {
    synSliceParams slice_params{};
    slice_params.ends[0] = input_shape[1]; // C
    slice_params.ends[1] = output_shape.at(2); // W
    slice_params.ends[2] = 1; // H
    slice_params.ends[3] = input_shape[0]; // N

    for (int i = self.dim(); i >= 0; --i) {
      slice_params.axes[i] = i;
      slice_params.starts[i] = 0;
      slice_params.steps[i] = 1;
    };
    std::vector<int64_t> slice_shape = {
        input_shape[0], // N
        static_cast<int64_t>(1), // H
        output_shape.at(2), // W
        input_shape[1]}; // C
    resize = BuildOp(
        graph,
        "slice",
        {resize[0].get()},
        {{slice_shape, ScalarType()}},
        &slice_params,
        sizeof(slice_params));
  };
  // Reshape N,H,W,C to N,W,C
  out_shape_3d = {input_shape[0], output_shape.at(2), input_shape[1]};
  auto reshape_nwc = BuildOp(
      graph, "reshape", {resize[0].get()}, {{out_shape_3d, ScalarType()}});

  // Transpose N,W,C to N,C,W
  auto transpose_ncw = BuildOp(
      graph,
      "transpose",
      {reshape_nwc[0].get()},
      {{output_shape, ScalarType(), is_output_persistent_list[0], true}},
      &trans_params,
      sizeof(trans_params));

  // output of transpose is the output of this op
  syn_out(0) = std::move(transpose_ncw.at(0));
}
} // namespace habana