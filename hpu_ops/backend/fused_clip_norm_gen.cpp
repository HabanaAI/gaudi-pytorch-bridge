/******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
  meta_vec.reserve(grads.size());

  for (const at::Tensor& grad : grads) {
    OutputMetaData meta;
    meta.dtype = grad.scalar_type();
    meta.shape = grad.sizes().vec();
    meta_vec.push_back(meta);
  }

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
    c10::ScalarType scalar_type) {
  ns_Reduction::ParamsV2 reduce_params{};
  reduce_params.reductionDimensionMask = 0;
  reduce_params.keepDim = 0;
  auto sum_result = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_square_multi_dim_fwd", scalar_type),
      {norm_input.syn_t},
      {{1, scalar_type}},
      &reduce_params,
      sizeof(reduce_params));

  auto norm = BuildOp(
      graph,
      get_guid_with_precision("sqrt_fwd", scalar_type),
      {sum_result.at(0).get()},
      {{1, scalar_type}});

  return norm;
}

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_total_norm(
    synapse_helpers::graph& graph,
    const std::vector<TensorsPair>& grads,
    c10::ScalarType scalar_type) {
  auto num_params = grads.size() - 1;
  int64_t scalar_shape[] = {1};
  double eps = 1e-6;

  // constant nodes.
  auto eps_ch =
      ConstantHelper(graph, static_cast<float>(eps), scalar_type, scalar_shape);
  auto zero_ch = ConstantHelper(graph, 0, scalar_type, scalar_shape);
  auto one_ch =
      ConstantHelper(graph, static_cast<float>(1.0), scalar_type, scalar_shape);

  std::vector<std::vector<synapse_helpers::tensor>> compute_norm_result;
  std::vector<synTensor> concat_inputs;
  for (size_t i = 0; i < num_params; ++i) {
    auto norm_result = compute_norm(graph, std::move(grads[i]), scalar_type);
    compute_norm_result.push_back(std::move(norm_result));
    concat_inputs.emplace_back(compute_norm_result.back().back().get());
  }

  synConcatenateParams concat_params{};
  concat_params.axis = 0;
  auto concat_op = BuildOp(
      graph,
      "concat",
      std::move(concat_inputs),
      {{num_params, scalar_type}},
      &concat_params,
      sizeof(concat_params));

  ns_Reduction::Params reduce_params{};
  reduce_params.reductionDimension = 0;
  auto sum_op = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_square_fwd", scalar_type),
      {concat_op.at(0).get()},
      {{1, scalar_type}},
      &reduce_params,
      sizeof(reduce_params));

  auto total_norm = BuildOp(
      graph,
      get_guid_with_precision("sqrt_fwd", scalar_type),
      {sum_op.at(0).get()},
      {{1, scalar_type}});

  return total_norm;
}

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_clip_coeff(
    synapse_helpers::graph& graph,
    const TensorsPair& max_norm,
    const synapse_helpers::tensor& total_norm,
    c10::ScalarType scalar_type) {
  int64_t scalar_shape[] = {1};
  double eps = 1e-6;

  // constant nodes.
  auto eps_ch =
      ConstantHelper(graph, static_cast<float>(eps), scalar_type, scalar_shape);
  auto one_ch =
      ConstantHelper(graph, static_cast<float>(1.0), scalar_type, scalar_shape);

  // total_norm + eps
  auto add_op = BuildOp(
      graph,
      get_guid_with_precision("add_fwd", scalar_type),
      {total_norm.get(), eps_ch.get()},
      {{1, scalar_type}});

  // clip_coef = max_norm / (total_norm + eps)
  auto clip_coef = BuildOp(
      graph,
      get_guid_with_precision("div_fwd", scalar_type),
      {max_norm.syn_t, add_op.at(0).get()},
      {{1, scalar_type}});

  auto clamp = BuildOp(
      graph,
      get_guid_with_precision("clamp_pt_fwd", scalar_type),
      {clip_coef.at(0).get(),
       nullptr,
       one_ch.get()}, // min = nullptr, max = 1.0
      {{1, scalar_type}});

  return clamp;
}

void FusedClipNormOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(stack, "FusedClipNormOp::AddNode");
  auto gradients = getNextInput<std::vector<TensorsPair>>(stackGetter);
  auto max_norm = getNextInput<TensorsPair>(stackGetter);
  auto norm_type = getNextInput<double>(stackGetter);
  auto scalar_type = gradients.at(0).pt_t.scalar_type();

  if (norm_type != 2)
    HABANA_ASSERT(0, "unsupported norm_type for FusedClipNorm");

  // gradients.back() returns a preallocated tensor to save the grads total
  // norm result. it must be the last element of grads list feeding this
  // operation thus -1 on num_params
  auto num_params = gradients.size() - 1;

  // clip_coeff = max_norm / (total_norm + eps)
  auto total_norm = compute_total_norm(graph, gradients, scalar_type);

  auto clip_coeff =
      compute_clip_coeff(graph, max_norm, total_norm.at(0), scalar_type);

  for (size_t i = 0; i < num_params; i++) {
    const auto& grad = gradients[i];

    auto mult_op = BuildOp(
        graph,
        get_guid_with_precision("mult_fwd", scalar_type),
        {clip_coeff.at(0).get(), grad.syn_t},
        {{grad.pt_t.sizes().vec(), scalar_type, i}});

    syn_out(i) = std::move(mult_op.at(0));
  }

  auto norm_identity = BuildIdentity(
      this, graph, total_norm.at(0).get(), {1}, scalar_type, num_params);

  // save total_norm result here on the last place of the input list.
  syn_out(num_params) = std::move(norm_identity);
}

} // namespace habana

static const auto& FusedClipNormKernelRegistry = habana::KernelRegistry().add(
    "hpu::fused_clip_norm",
    KERNEL_FN_GLOBAL(habana::FusedClipNormOp));
