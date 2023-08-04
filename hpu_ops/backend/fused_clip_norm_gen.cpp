/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

#include "hpu_ops/fused_clip_norm.h"

#include "habana_helpers/logging.h"

#include "hpu_ops/backend/reduction_template.h"

namespace habana {

OutputMetaDataVector FusedClipNormOp::FusedClipNormMeta(
    const at::Stack& stack) {
  PT_OP_INFO("fused_clip_norm :", "FusedClipNormMeta");
  OutputMetaDataVector meta_vec;

  auto grads = stack[0].toTensorList();
  meta_vec.reserve(grads.size() + 1);

  for (const at::Tensor& grad : grads) {
    OutputMetaData meta;
    meta.dtype = grad.scalar_type();
    meta.shape = grad.sizes().vec();
    meta_vec.push_back(meta);
  }

  // additional 1-dim size tensor for total_norm result
  OutputMetaData meta;
  const at::Tensor& grad = grads[0];
  meta.dtype = grad.scalar_type();
  meta.shape = {1};
  meta_vec.push_back(meta);

  return meta_vec;
}

FusedClipNormOp::FusedClipNormOp(int device_id, c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "fused_clip_norm",
          scalar_type,
          {},
          {0}, // inplace id
          {},
          false) {
  PT_OP_INFO("fused_clip_norm :", "FusedClipNormOp constructor");

  SetOutputMetaFn(FusedClipNormMeta);
}

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_norm(
    synapse_helpers::graph& graph,
    const TensorsPair& norm_input,
    const double norm_type) {
  auto input_shape = norm_input.pt_t.sizes();
  auto out_shape = input_shape;
  auto n_dims = norm_input.pt_t.dim();

  if (n_dims <= 1 || out_shape[0] == 1) {
    auto identity_op = BuildIdentity(
        this, graph, norm_input.syn_t, input_shape.vec(), ScalarType());

    auto mult_op = BuildOp(
        graph,
        get_guid_with_precision("mult", ScalarType()),
        {identity_op.get(), norm_input.syn_t},
        {{out_shape, ScalarType()}});

    ns_Reduction::Params reduce_params{};
    reduce_params.reductionDimension = 0;
    auto sum_result = BuildOp(
        graph,
        get_guid_with_precision("reduce_sum_fwd", ScalarType()),
        {mult_op.at(0).get()},
        {{1, ScalarType()}},
        &reduce_params,
        sizeof(reduce_params));

    auto my_out = BuildOp(
        graph,
        get_guid_with_precision("sqrt_fwd", ScalarType()),
        {sum_result.at(0).get()},
        {{1, ScalarType()}});

    return std::move(my_out);
  } else {
    auto norm = BuildOp(
        graph,
        get_guid_with_precision("frobenius_norm_fwd", ScalarType()),
        {norm_input.syn_t},
        {{1, ScalarType()}});

    return std::move(norm);
  }
}

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_clip_coeff(
    synapse_helpers::graph& graph,
    const std::vector<TensorsPair>& grads,
    const TensorsPair& max_norm) {
  auto num_params = grads.size() - 1;
  int64_t scalar_shape[] = {1};
  double norm_type =
      2.0; // assert on this value is performed earlier in the call stack.
  double eps = 1e-6;

  // constant nodes.
  auto eps_ch = ConstantHelper(
      graph, static_cast<float>(eps), ScalarType(), scalar_shape);
  auto zero_ch = ConstantHelper(graph, 0, ScalarType(), scalar_shape);

  std::vector<std::vector<synapse_helpers::tensor>> compute_norm_result;
  std::vector<synTensor> concat_inputs;
  for (size_t i = 0; i < num_params; ++i) {
    auto norm_result = compute_norm(graph, grads[i], norm_type);
    compute_norm_result.push_back(std::move(norm_result));
    concat_inputs.emplace_back(compute_norm_result.back().back().get());
  }

  synConcatenateParams concat_params{};
  concat_params.axis = 0;
  auto concat_op = BuildOp(
      graph,
      "concat",
      concat_inputs,
      {{num_params, ScalarType()}},
      &concat_params,
      sizeof(concat_params));

  auto identity_op = BuildIdentity(
      this, graph, concat_op.at(0).get(), num_params, ScalarType());

  // the next three ops with compute total_norm.
  auto mult_op = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {identity_op.get(), concat_op.at(0).get()},
      {{num_params, ScalarType()}});

  ns_Reduction::Params reduce_params{};
  reduce_params.reductionDimension = 0;
  auto sum_op = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_fwd", ScalarType()),
      {mult_op.at(0).get()},
      {{1, ScalarType()}},
      &reduce_params,
      sizeof(reduce_params));

  auto sqrt_op = BuildOp(
      graph,
      get_guid_with_precision("sqrt_fwd", ScalarType()),
      {sum_op.at(0).get()},
      {{1, ScalarType(), num_params}});

  // save total_norm result here on the last place of the input list.
  syn_out(num_params) = std::move(sqrt_op.at(0));

  // total_norm + eps
  auto add_op = BuildOp(
      graph,
      get_guid_with_precision("add_fwd", ScalarType()),
      {syn_out(num_params).get(), eps_ch.get()},
      {{1, ScalarType()}});

  // clip_coef = max_norm / (total_norm + eps)
  auto div_op = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", ScalarType()),
      {max_norm.syn_t, add_op.at(0).get()},
      {{1, ScalarType()}});

  // mask = total_norm > max_norm
  auto ge_1 = BuildOp(
      graph,
      get_guid_with_precision("greater_fwd", ScalarType()),
      {syn_out(num_params).get(), max_norm.syn_t},
      {{1, ScalarType()}});

  // cast bool to float
  auto cast_i8_to_f32_1 = CastHelper(
      graph, ge_1[0].get(), 1, c10::ScalarType::Bool, c10::ScalarType::Float);

  // mask * clip_coef
  auto mult_op_2 = BuildOp(
      graph,
      get_guid_with_precision("mult", ScalarType()),
      {cast_i8_to_f32_1.get(), div_op.at(0).get()},
      {{1, ScalarType()}});

  // imask = (mask == 0)
  auto eq_1 = BuildOp(
      graph,
      get_guid_with_precision("equal_fwd", ScalarType()),
      {cast_i8_to_f32_1.get(), zero_ch.get()},
      {{1, ScalarType()}});

  // cast bool to float
  auto cast_i8_to_f32_2 = CastHelper(
      graph, eq_1[0].get(), 1, c10::ScalarType::Bool, c10::ScalarType::Float);

  // mask * clip_coef + imask
  auto add_op_2 = BuildOp(
      graph,
      get_guid_with_precision("add_fwd", ScalarType()),
      {mult_op_2.at(0).get(), cast_i8_to_f32_2.get()},
      {{1, ScalarType()}});

  synSliceParamsV2 slice_params{};
  slice_params.axes[0] = 0;
  slice_params.starts[0] = 0;
  slice_params.ends[0] = 1;
  slice_params.steps[0] = 1;

  // take just the first element of add since all values would be repeated
  auto slice = BuildOp(
      graph,
      get_guid_with_precision("slice", ScalarType()),
      {add_op_2.at(0).get()},
      {{1, ScalarType()}},
      &slice_params,
      sizeof(slice_params));

  return std::move(slice);
}

void FusedClipNormOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "FusedClipNormOp::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto max_norm = getNextInput<TensorsPair>(stackGetter);
  auto norm_type = getNextInput<double>(stackGetter);

  if (norm_type != 2)
    HABANA_ASSERT(0, "unsupported norm_type for FusedClipNorm");

  // gradients.back() returns a preallocated tensor to save the grads total norm
  // result. it must be the last element of grads list feeding this operation
  // thus -1 on num_params
  auto num_params = gradients.size() - 1;

  // clip_coeff = max_norm / (total_norm + eps)
  auto clip_coeff = compute_clip_coeff(graph, gradients, max_norm);

  for (size_t i = 0; i < num_params; i++) {
    // grad * clip_coef
    auto mult_op = BuildOp(
        graph,
        get_guid_with_precision("mult", ScalarType()),
        {gradients.at(i).syn_t, clip_coeff.at(0).get()},
        {{gradients.at(i).pt_t.sizes().vec(), ScalarType(), i}});

    syn_out(i) = std::move(mult_op.at(0));
  }
}

} // namespace habana

static const auto& FusedClipNormKernelRegistry = habana::KernelRegistry().add(
    "hpu::fused_clip_norm",
    KERNEL_FN_GLOBAL(habana::FusedClipNormOp));