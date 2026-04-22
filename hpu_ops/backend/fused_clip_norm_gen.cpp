/**
 * Copyright (c) 2021-2026 Intel Corporation
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

#include <shared_layer_api.hpp>
#include "hpu_ops/fused_clip_norm.h"

#include "common/warning_suppress.h"
#include "habana_helpers/logging.h"

namespace habana {

OutputMetaDataVector FusedClipNormOp::FusedClipNormMeta(
    const at::Stack& stack) {
  PT_OP_INFO("fused_clip_norm :", "FusedClipNormMeta");
  OutputMetaDataVector meta_vec;

  auto grads = stack[0].toTensorList();
  meta_vec.reserve(grads.size());
  SUPPRESS_WDANGLING_REFERENCE(for (const at::Tensor& grad : grads)) {
    auto& meta = meta_vec.emplace_back();
    meta.dtype = grad.scalar_type();
    meta.shape = grad.sizes().vec();
  }

  return meta_vec;
}

SharedMetaDataVector FusedClipNormSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& gradients = stack.at(0).toTensorVector();
  const auto& max_norm = stack.at(1).toTensor();
  const auto precision_type = gradients[0].scalar_type();

  const auto num_params = gradients.size() - 1;
  SharedMetaTensor constant_tensor{1, precision_type};
  SharedMetaDataVector shared_meta_vec;
  shared_meta_vec.reserve(7 + (2 * num_params));

  SharedMetaVector concat_inputs;
  for (size_t i = 0; i < num_params && i < SharedLayer::MAX_TENSOR_NR; i++) {
    auto& sum_shared_meta =
        shared_meta_vec.emplace_back("reduce_sum_square_multi_dim_fwd");
    sum_shared_meta.inputs_data = {{gradients[i].dim(), precision_type}};
    sum_shared_meta.outputs_data = {constant_tensor};

    auto& norm_shared_meta = shared_meta_vec.emplace_back("sqrt_fwd");
    norm_shared_meta.inputs_data = sum_shared_meta.outputs_data;
    norm_shared_meta.outputs_data = {constant_tensor};
    concat_inputs.push_back(constant_tensor);
  }

  auto& concat_shared_meta = shared_meta_vec.emplace_back("concat");
  concat_shared_meta.inputs_data = concat_inputs;
  concat_shared_meta.outputs_data = {constant_tensor};

  auto& sum_shared_meta = shared_meta_vec.emplace_back("reduce_sum_square_fwd");
  sum_shared_meta.inputs_data = concat_shared_meta.outputs_data;
  sum_shared_meta.outputs_data = sum_shared_meta.inputs_data;

  auto& total_norm_shared_meta = shared_meta_vec.emplace_back("sqrt_fwd");
  total_norm_shared_meta.inputs_data = sum_shared_meta.outputs_data;
  total_norm_shared_meta.outputs_data = total_norm_shared_meta.inputs_data;

  auto& add_shared_meta = shared_meta_vec.emplace_back("add_fwd");
  add_shared_meta.inputs_data = {
      total_norm_shared_meta.outputs_data[0], constant_tensor};
  add_shared_meta.outputs_data = {constant_tensor};

  auto& clip_coef_shared_meta = shared_meta_vec.emplace_back("div_fwd");
  clip_coef_shared_meta.inputs_data = {
      {max_norm.dim(), precision_type}, add_shared_meta.outputs_data[0]};
  clip_coef_shared_meta.outputs_data = {constant_tensor};

  auto& clamp_shared_meta = shared_meta_vec.emplace_back("clamp_pt_fwd");
  clamp_shared_meta.inputs_data = {
      clip_coef_shared_meta.outputs_data[0],
      createOptionalNotPresentSharedMetaTensor(),
      constant_tensor};
  clamp_shared_meta.outputs_data = {constant_tensor};

  for (size_t i = 0; i < num_params; i++) {
    auto& mult_shared_meta = shared_meta_vec.emplace_back("mult_fwd");
    mult_shared_meta.inputs_data = {
        clamp_shared_meta.outputs_data[0],
        {gradients[i].dim(), precision_type}};
    mult_shared_meta.outputs_data.emplace_back(
        gradients[i].dim(), precision_type);
  }
  return shared_meta_vec;
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

using namespace std::literals;

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_norm(
    synapse_helpers::graph& graph,
    const TensorsPair& norm_input,
    c10::ScalarType scalar_type) {
  ns_Reduction::ParamsV2 reduce_params{};
  reduce_params.reductionDimensionMask = 0;
  reduce_params.keepDim = false;
  auto sum_result = BuildOp(
      graph,
      get_guid_with_precision("reduce_sum_square_multi_dim_fwd"sv, scalar_type),
      {norm_input.syn_t},
      {{1, scalar_type}},
      &reduce_params,
      sizeof(reduce_params));

  auto norm = BuildOp(
      graph,
      get_guid_with_precision("sqrt_fwd"sv, scalar_type),
      {sum_result.at(0).get()},
      {{1, scalar_type}});

  return norm;
}

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_total_norm(
    synapse_helpers::graph& graph,
    const std::vector<TensorsPair>& grads,
    c10::ScalarType scalar_type) {
  auto num_params = grads.size() - 1;
  int64_t scalar_shape = 1L;
  const float eps = 1e-6F;
  const at::IntArrayRef scalar_shape_ref = c10::makeArrayRef(&scalar_shape, 1);

  // constant nodes.
  auto eps_ch = ConstantHelper(graph, eps, scalar_type, scalar_shape_ref);
  auto zero_ch = ConstantHelper(graph, 0, scalar_type, scalar_shape_ref);
  auto one_ch = ConstantHelper(graph, 1.0F, scalar_type, scalar_shape_ref);

  std::vector<std::vector<synapse_helpers::tensor>> compute_norm_result;
  std::vector<synTensor> concat_inputs;
  for (size_t i = 0; i < num_params; ++i) {
    auto norm_result = compute_norm(graph, grads[i], scalar_type);
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
      get_guid_with_precision("reduce_sum_square_fwd"sv, scalar_type),
      {concat_op.at(0).get()},
      {{1, scalar_type}},
      &reduce_params,
      sizeof(reduce_params));

  auto total_norm = BuildOp(
      graph,
      get_guid_with_precision("sqrt_fwd"sv, scalar_type),
      {sum_op.at(0).get()},
      {{1, scalar_type}});

  return total_norm;
}

std::vector<synapse_helpers::tensor> FusedClipNormOp::compute_clip_coeff(
    synapse_helpers::graph& graph,
    const TensorsPair& max_norm,
    const synapse_helpers::tensor& total_norm,
    c10::ScalarType scalar_type) {
  int64_t scalar_shape = 1L;
  constexpr float eps = 1e-6F;
  const at::IntArrayRef scalar_shape_ref = c10::makeArrayRef(&scalar_shape, 1);

  // constant nodes.
  auto eps_ch = ConstantHelper(graph, eps, scalar_type, scalar_shape_ref);
  auto one_ch = ConstantHelper(graph, 1.0F, scalar_type, scalar_shape_ref);

  // total_norm + eps
  auto add_op = BuildOp(
      graph,
      get_guid_with_precision("add_fwd"sv, scalar_type),
      {total_norm.get(), eps_ch.get()},
      {{1, scalar_type}});

  // clip_coef = max_norm / (total_norm + eps)
  auto clip_coef = BuildOp(
      graph,
      get_guid_with_precision("div_fwd"sv, scalar_type),
      {max_norm.syn_t, add_op.at(0).get()},
      {{1, scalar_type}});

  auto clamp = BuildOp(
      graph,
      get_guid_with_precision("clamp_pt_fwd"sv, scalar_type),
      {clip_coef.at(0).get(),
       nullptr,
       one_ch.get()}, // min = nullptr, max = 1.0
      {{1, scalar_type}});

  return clamp;
}

void FusedClipNormOp::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "FusedClipNormOp::AddNode");
  auto gradients = stackGetter.getNextInput<std::vector<TensorsPair>>();
  auto max_norm = stackGetter.getNextInput<TensorsPair>();
  auto norm_type = stackGetter.getNextInput<double>();
  auto scalar_type = gradients.at(0).pt_t.scalar_type();
  if (norm_type != 2) {
    HABANA_ASSERT(0, "unsupported norm_type for FusedClipNorm");
  }

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
        get_guid_with_precision("mult_fwd"sv, scalar_type),
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

static const auto& FusedClipNormKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::fused_clip_norm",
        habana::FusedClipNormOp);
