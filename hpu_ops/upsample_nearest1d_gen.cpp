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

namespace habana {

sizes_vec UpsampleNearest1DFwdOutputShape(const at::Stack& stack, bool) {
  auto self = stack.at(0).toTensor();
  auto size = stack.at(1);
  auto scale = stack.at(2);
  std::vector<int64_t> output_shape;

  TORCH_CHECK(
      self.dim() == 3,
      "UpsampleNearest is expected input_size equals to 3, but got size ",
      self.dim());

  TORCH_CHECK(
      !(size == c10::nullopt && scale == c10::nullopt),
      "Must specify exactly one of output_size and scale_factors");

  if (!size.isNone()) {
    TORCH_CHECK(
        size.toIntVector().empty() == false,
        "UpsampleNearest is expected output_size equals to 1, but got size 0",
        size.toIntVector().size());
    output_shape = {self.sizes()[0], self.sizes()[1], size.toIntVector().at(0)};
  } else if (!scale.isNone()) {
    double scale_factor =
        scale.isDouble() ? scale.toDouble() : scale.toDoubleVector().at(0);
    auto width = self.sizes()[2];
    output_shape = {
        self.sizes()[0],
        self.sizes()[1],
        static_cast<int64_t>(width * scale_factor)};
  }

  // input_width and output_width should be greater than 0
  TORCH_CHECK(
      self.sizes()[2] > 0 && output_shape.at(2) > 0,
      "Input and output sizes should be greater than 0, but got input (W: ",
      self.sizes()[2],
      ") and output (W: ",
      output_shape.at(2),
      ")");
  return {output_shape};
}

sizes_vec UpsampleNearest1DBwdOutputShape(const at::Stack& stack, bool) {
  std::vector<int64_t> output_shape = stack.at(2).toIntList().vec();
  return {output_shape};
}

std::shared_ptr<void> FillUpsampleNearest1DParams(
    const at::Stack& stack,
    size_t& size,
    bool is_forward) {
  PARAMS_STUB(ns_ResizeKernel::Params);
  auto out_size = stack.at(1);
  auto scales = is_forward ? stack.at(2) : stack.at(3);
  params->mode = ResizeInterpolationMode_t::RESIZE_INTER_NEAREST;
  params->nearestMode = ResizeNearestMode_t::FLOOR;
  params->coordTransMode =
      ResizeCoordinateTransformationMode_t::ASYMMETRIC_MODE;
  if (!out_size.isNone()) {
    params->useScales = false;
    params->size1 = out_size.toIntVector().at(0);
  }
  if (!scales.isNone()) {
    params->useScales = true;
    params->scaleDim1 =
        scales.isDouble() ? scales.toDouble() : scales.toDoubleVector().at(0);
    params->scaleDim2 = 1.0;
    params->scaleDim3 = 1.0;
  }
  return params;
}

void UpsampleNearest1DFwd::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const torch::Tensor& self = stack.at(0).toTensor();
  auto input_shape = self.sizes();
  auto out_size = stack.at(1);
  auto scales = stack.at(2);
  auto outshape = UpsampleNearest1DFwdOutputShape(stack)[0];

  // Transpose params
  synTransposeParams trans_params{};
  trans_params.tensorDim = self.dim();
  for (int i = 0; i < self.dim(); ++i) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  // Transpose N,C,W to N,W,C
  std::vector<int64_t> outshape_3d{
      input_shape[0], input_shape[2], input_shape[1]};

  auto transpose_nwc = BuildOp(
      graph,
      "transpose",
      {syn_in(0)},
      {{outshape_3d, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  // Reshape N,W,C to N,H,W,C
  std::vector<int64_t> outshape_4d{
      input_shape[0], static_cast<int64_t>(1), input_shape[2], input_shape[1]};

  auto reshape_nhwc = BuildOp(
      graph,
      "reshape",
      {transpose_nwc[0].get()},
      {{outshape_4d, ScalarType()}});

  size_t size = 0;
  bool is_forward = true;
  const auto& resize_params =
      FillUpsampleNearest1DParams(stack, size, is_forward);

  // modify input width value with output width value
  outshape_4d.at(2) = (!out_size.isNone() && !scales.isNone())
      ? static_cast<int64_t>(input_shape[2] * scales.toDouble())
      : outshape.at(2);

  auto resize = BuildOp(
      graph,
      "resize_fwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {reshape_nhwc[0].get()},
      {{outshape_4d, ScalarType()}},
      resize_params.get(),
      size);

  // Slice the resize result when both size and scale is provided
  if (!out_size.isNone() && !scales.isNone()) {
    synSliceParams slice_params{};
    slice_params.ends[0] = input_shape[1];
    slice_params.ends[1] = outshape.at(2);
    slice_params.ends[2] = 1;
    slice_params.ends[3] = input_shape[0];
    for (int i = self.dim(); i >= 0; --i) {
      slice_params.axes[i] = i;
      slice_params.starts[i] = 0;
      slice_params.steps[i] = 1;
    };
    std::vector<int64_t> slice_shape = {
        input_shape[0], 1, outshape.at(2), input_shape[1]};
    resize = BuildOp(
        graph,
        "slice",
        {resize[0].get()},
        {{slice_shape, ScalarType()}},
        &slice_params,
        sizeof(slice_params));
  };

  // Reshape N,H,W,C to N,W,C
  outshape_3d = {input_shape[0], outshape.at(2), input_shape[1]};

  auto reshape_nwc = BuildOp(
      graph, "reshape", {resize[0].get()}, {{outshape_3d, ScalarType()}});

  // Transpose N,W,C to N,C,W
  auto transpose_ncw = BuildOp(
      graph,
      "transpose",
      {reshape_nwc[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}},
      &trans_params,
      sizeof(trans_params));

  // output of transpose is the output of this op
  syn_out(0) = std::move(transpose_ncw[0]);
}

void UpsampleNearest1DBwd::AddNode(
    synapse_helpers::graph& graph,
    at::Stack& stack,
    const std::vector<bool>& is_output_persistent_list) {
  const torch::Tensor& self = stack.at(0).toTensor();
  auto input_shape = self.sizes();
  auto outshape = UpsampleNearest1DBwdOutputShape(stack)[0];

  // Transpose params
  synTransposeParams trans_params{};
  trans_params.tensorDim = self.dim();
  for (int i = 0; i < self.dim(); i++) {
    trans_params.permutation[i] = static_cast<TransposePermutationDim>(i);
  }
  std::swap(trans_params.permutation[0], trans_params.permutation[1]);

  // Transpose N,C,W to N,W,C
  std::vector<int64_t> outshape_3d{
      input_shape[0], input_shape[2], input_shape[1]};

  auto transpose_nwc = BuildOp(
      graph,
      "transpose",
      {syn_in(0)},
      {{outshape_3d, ScalarType()}},
      &trans_params,
      sizeof(trans_params));

  // Reshape N,W,C to N,H,W,C
  std::vector<int64_t> outshape_4d{
      input_shape[0], static_cast<int64_t>(1), input_shape[2], input_shape[1]};

  auto reshape_nhwc = BuildOp(
      graph,
      "reshape",
      {transpose_nwc[0].get()},
      {{outshape_4d, ScalarType()}});

  // Resize
  size_t size = 0;
  bool is_forward = false;
  const auto& resize_params =
      FillUpsampleNearest1DParams(stack, size, is_forward);

  // modify input width value with output width value
  outshape_4d.at(2) = outshape.at(2);

  auto resize = BuildOp(
      graph,
      "resize_bwd_" + habana_helpers::name_suffix_from_type(ScalarType()),
      {reshape_nhwc[0].get()},
      {{outshape_4d, ScalarType()}},
      resize_params.get(),
      size);

  // Reshape N,H,W,C to N,W,C
  auto outshape_nwc = {input_shape[0], outshape.at(2), input_shape[1]};

  auto reshape_nwc = BuildOp(
      graph, "reshape", {resize[0].get()}, {{outshape_nwc, ScalarType()}});

  // Transpose N,W,C to N,C,W
  auto transpose_ncw = BuildOp(
      graph,
      "transpose",
      {reshape_nwc[0].get()},
      {{outshape, ScalarType(), is_output_persistent_list[0], true}},
      &trans_params,
      sizeof(trans_params));

  // output of transpose is the output of this op
  syn_out(0) = std::move(transpose_ncw[0]);
}
} // namespace habana