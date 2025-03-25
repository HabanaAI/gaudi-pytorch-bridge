/**
 * Copyright (c) 2024-2025 Intel Corporation
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

#include "generated/backend/mixture_of_experts.h"
#include "backend/habana_device/HPUGuardImpl.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "hpu_ops/custom_op_outshape.h"
#include "hpu_ops/mixture_of_experts.h"

namespace sh = synapse_helpers;

namespace habana {

template <class DimT>
DimT ceil_div(DimT a, DimT b) {
  return (a + b - 1) / b;
}

template <class DimT>
inline DimT get_split_factor(DimT num_experts) {
  DimT split_factor = 8;
  if (num_experts == 1) {
    split_factor = 64;
  } else if (num_experts >= 2 && num_experts < 4) {
    split_factor = 32;
  } else if (num_experts >= 4 && num_experts < 8) {
    split_factor = 16;
  }
  return split_factor;
}

template <class DimT>
DimT calculate_tokens_per_chunk(
    DimT num_tokens,
    DimT experts_per_token,
    DimT num_experts,
    DimT device_threshold,
    DimT num_chunks,
    DimT split_factor) {
  DimT tokens_per_chunk;
  if (num_chunks / num_experts <= split_factor) {
    tokens_per_chunk = device_threshold;
  } else {
    tokens_per_chunk =
        ceil_div(num_chunks, num_experts * split_factor) * device_threshold;
  }
  DimT minCS = std::min(
      ceil_div(
          experts_per_token * num_tokens, 2 * num_experts * device_threshold) *
          device_threshold,
      4 * device_threshold);
  if (tokens_per_chunk < minCS) {
    tokens_per_chunk = minCS;
  }
  return tokens_per_chunk;
}

template <class DimT>
DimT calculate_num_chunks(
    DimT num_tokens,
    DimT experts_per_token,
    DimT num_experts,
    DimT device_threshold,
    DimT tokens_per_chunk) {
  if (num_tokens <= device_threshold) {
    return std::min(experts_per_token * num_tokens, num_experts);
  } else {
    return num_experts *
        (ceil_div(
             experts_per_token * num_tokens, num_experts * tokens_per_chunk) +
         1);
  }
}

template <class DimT>
std::pair<DimT, DimT> calculate_tokens_per_chunk_and_num_chunks(
    DimT num_tokens,
    DimT experts_per_token,
    DimT num_experts) {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  const bool isGaudi3 =
      habana::HPUDeviceContext::get_device().type() == synDeviceGaudi3;

  DimT device_threshold = isGaudi3 ? 512 : 256;
  DimT split_factor = get_split_factor(num_experts);

  // First calculate num_chunks using tokens_per_chunk=device_threshold, then
  // recalculate tokens_per_chunk and num_chunks.
  DimT tokens_per_chunk = device_threshold;
  DimT num_chunks = calculate_num_chunks(
      num_tokens,
      experts_per_token,
      num_experts,
      device_threshold,
      tokens_per_chunk);

  tokens_per_chunk = calculate_tokens_per_chunk(
      num_tokens,
      experts_per_token,
      num_experts,
      device_threshold,
      num_chunks,
      split_factor);
  num_chunks = calculate_num_chunks(
      num_tokens,
      experts_per_token,
      num_experts,
      device_threshold,
      tokens_per_chunk);

  return std::make_pair(tokens_per_chunk, num_chunks);
}

template <class DimT>
sizes_vec_template<DimT> MixtureOfExpertsFwdSizes(
    c10::ArrayRef<DimT> hidden_state_sizes,
    c10::ArrayRef<DimT> expert_routing_table_sizes,
    DimT num_experts,
    DimT H2,
    bool fused_weights) {
  DimT num_tokens = hidden_state_sizes[0];
  DimT H = hidden_state_sizes[1];
  DimT experts_per_token = expert_routing_table_sizes[1];

  auto [tokens_per_chunk, num_chunks] =
      calculate_tokens_per_chunk_and_num_chunks(
          num_tokens, experts_per_token, num_experts);

  sizes_vec_template<DimT> results = {
      {num_tokens, H},
      {num_chunks, tokens_per_chunk, H},
      {num_tokens, experts_per_token},
      {num_tokens, experts_per_token},
      {num_chunks},
      {num_chunks, tokens_per_chunk},
      {num_chunks, tokens_per_chunk, H2}};
  if (fused_weights) {
    H2 = H2 / 2;
  }

  results.push_back({num_chunks, tokens_per_chunk, H2});
  results.push_back({num_chunks, tokens_per_chunk, H2});
  if (!fused_weights) {
    results.push_back({num_chunks, tokens_per_chunk, H2});
  }
  results.push_back({num_chunks, tokens_per_chunk, H});

  return results;
}

std::vector<std::vector<int64_t>> MixtureOfExpertsFwdShapes(
    const at::Stack& stack) {
  const bool fused_weights = !stack.at(5).isTensorList();

  const size_t permute_weights_idx = fused_weights ? 5 : 6;
  const bool permuted = stack.at(permute_weights_idx).toBool();
  const auto& weights_shape = stack.at(3).toTensorList().get(0).sizes();

  return MixtureOfExpertsFwdSizes(
      stack_tensor(stack, 0).sizes(),
      stack_tensor(stack, 1).sizes(),
      static_cast<int64_t>(stack.at(3).toTensorList().size()),
      static_cast<int64_t>(weights_shape[permuted ? 0 : 1]),
      fused_weights);
}

sym_sizes_vec mixture_of_experts_fwd_out_shape(
    const std::vector<at::Tensor>& inputs,
    const std::vector<int64_t>& params) {
  HABANA_ASSERT(inputs.size() == 2);
  HABANA_ASSERT(params.size() == 3);

  const auto& hidden_state_sizes = inputs[0].sym_sizes();
  const auto& expert_routing_table_sizes = inputs[1].sym_sizes();

  return MixtureOfExpertsFwdSizes(
      hidden_state_sizes,
      expert_routing_table_sizes,
      c10::SymInt(params[0]),
      c10::SymInt(params[1]),
      params[2]);
}

REGISTER_CUSTOM_OP_OUTSHAPE_FUN(
    mixture_of_experts_fwd,
    mixture_of_experts_fwd_out_shape);

OutputMetaDataVector MixtureOfExpertsFp8Meta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.shape = stack_tensor(stack, 0).sizes().vec();
  meta.dtype = c10::ScalarType::BFloat16;
  return {meta};
}

MixtureOfExperts::MixtureOfExperts(
    int device_id,
    c10::ScalarType scalar_type,
    bool measurement_mode)
    : OpBackend(
          device_id,
          "moe",
          scalar_type,
          measurement_mode ? std::vector<int>{0, 0} : std::vector<int>{0},
          {},
          {},
          false),
      measurement_mode(measurement_mode) {}

static const std::map<std::string_view, MoeActivationMode_t> activationModeMap =
    {{"gelu", MoeActivationMode_t::MOE_ACTIVATION_MODE_GELU},
     {"relu", MoeActivationMode_t::MOE_ACTIVATION_MODE_RELU},
     {"silu", MoeActivationMode_t::MOE_ACTIVATION_MODE_SILU}};

std::shared_ptr<void> FillMixtureOfExpertsParams(
    const at::Stack& stack,
    size_t& size,
    const int permuted_weights_idx,
    const bool fused_gemm,
    const bool measurement_mode,
    const bool dynamic_scale) {
  const auto permuted_weights = stack.at(permuted_weights_idx).toBool();
  const auto activation_mode =
      stack.at(permuted_weights_idx + 1).to<std::string_view>();
  auto activationIterator = activationModeMap.find(activation_mode);
  HABANA_ASSERT(
      activationIterator != activationModeMap.end(),
      "Activation \"",
      activation_mode,
      "\" not found among MoeActivationMode_t enum values.")

  PARAMS_STUB(ns_MoeKernel::ParamsV2);
  params->experts.activation = activationIterator->second;
  params->router.experts_min =
      stack.at(permuted_weights_idx + 2).toScalar().toInt();
  params->router.experts_max =
      stack.at(permuted_weights_idx + 3).toScalar().toInt();
  params->flags = permuted_weights ? MoeFlags_t::MOE_FLAGS_PERMUTED_WEIGHTS : 0;
  params->flags |= (fused_gemm ? MoeFlags_t::MOE_FLAGS_FUSED_GEMM : 0);
  params->flags |= (measurement_mode ? MoeFlags_t::MOE_FLAGS_CALC_AMAX : 0);
  params->flags |= (dynamic_scale ? MoeFlags_t::MOE_FLAGS_DYNAMIC_SCALE : 0);
  return params;
}

OutputMetaDataVector MixtureOfExpertsMeta(
    const at::Stack& stack,
    bool measurement_mode) {
  const auto& self = stack_tensor(stack, 0);
  OutputMetaDataVector meta(1);
  meta[0].shape = self.sizes().vec();
  meta[0].dtype = self.scalar_type();
  if (measurement_mode) {
    auto numExperts = stack.at(3).toTensorList().size();
    OutputMetaData measurement_meta;
    measurement_meta.shape = {static_cast<int>(numExperts)};
    measurement_meta.dtype = c10::ScalarType::Float;
    meta.push_back(measurement_meta);
  }
  return meta;
}

std::vector<NodeAttr::NodeOutputAttr> createOutputAttrs(
    const OutputMetaDataVector& meta) {
  std::vector<NodeAttr::NodeOutputAttr> output_attrs;
  for (size_t i = 0; i < meta.size(); ++i) {
    output_attrs.push_back({meta[i].shape, meta[i].dtype, i});
  }
  return output_attrs;
}

OutputMetaDataVector MixtureOfExpertsFwdMeta(const at::Stack& stack) {
  at::ScalarType self_type = stack_tensor(stack, 0).scalar_type();

  std::vector<std::vector<int64_t>> output_shapes =
      MixtureOfExpertsFwdShapes(stack);
  std::vector<at::ScalarType> dtypes = {
      self_type,
      self_type,
      torch::kInt,
      torch::kInt,
      torch::kInt,
      self_type,
      self_type,
      self_type,
      self_type,
      self_type,
      self_type};

  const size_t output_number = output_shapes.size();
  OutputMetaDataVector meta(output_number);
  for (size_t i = 0; i < output_number; ++i) {
    meta[i].shape = output_shapes[i];
    meta[i].dtype = dtypes[i];
  }
  return meta;
}

OutputMetaDataVector MixtureOfExpertsFwdRecompMeta(const at::Stack& stack) {
  const auto& self = stack_tensor(stack, 0);
  return {{self.scalar_type(), self.sizes().vec()}};
}

OutputMetaDataVector MixtureOfExpertsBwdMeta(const at::Stack& stack) {
  const auto& grad = stack_tensor(stack, 0);

  const bool is_recompute = stack.size() < 17;
  const bool fused_weights =
      is_recompute ? !stack.at(6).isTensorList() : !stack.at(12).isTensorList();
  const size_t first_weights_index = is_recompute ? 4 : fused_weights ? 10 : 11;
  const size_t num_experts =
      stack.at(first_weights_index).toTensorList().size();
  const size_t weights_per_expert = fused_weights ? 2 : 3;

  const size_t router_weights_shape_idx = fused_weights ? 16 : 18;
  std::vector<int64_t> router_weights_shape = is_recompute
      ? stack_tensor(stack, 3).sizes().vec()
      : stack.at(router_weights_shape_idx).toIntVector();

  OutputMetaDataVector meta(2 + num_experts * weights_per_expert);
  meta[0].shape = grad.sizes().vec();
  meta[0].dtype = grad.scalar_type();

  meta[1].shape = router_weights_shape;
  meta[1].dtype = grad.scalar_type();

  const auto& w1 = stack.at(first_weights_index).toTensorList();
  for (size_t i = 0; i < num_experts; ++i) {
    meta[i + 2].shape = w1[i].sizes().vec();
    meta[i + 2].dtype = w1[i].scalar_type();
  }

  const auto& w2 = stack.at(first_weights_index + 1).toTensorList();
  for (size_t i = 0; i < num_experts; ++i) {
    meta[i + 2 + num_experts].shape = w2[i].sizes().vec();
    meta[i + 2 + num_experts].dtype = w2[i].scalar_type();
  }

  if (!fused_weights) {
    const auto& w3 = stack.at(first_weights_index + 2).toTensorList();
    for (size_t i = 0; i < num_experts; ++i) {
      meta[i + 2 + 2 * num_experts].shape = w3[i].sizes().vec();
      meta[i + 2 + 2 * num_experts].dtype = w3[i].scalar_type();
    }
  }

  return meta;
}

using namespace std::literals;

void MixtureOfExperts::AddNode(sh::graph& graph, const at::Stack& stack) {
  const bool fused_weights = !stack.at(5).isTensorList();
  auto num_experts = stack.at(3).toTensorList().size();
  auto weights_per_expert = fused_weights ? 2 : 3;
  auto permute_weights_idx = fused_weights ? 5 : 6;
  size_t expected_stack_size = fused_weights ? 10 : 11;

  const bool measurement_mode =
      this->measurement_mode && stack.size() == expected_stack_size;
  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 3 + num_experts * weights_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, measurement_mode, false);
  auto meta = MixtureOfExpertsMeta(stack, measurement_mode);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs{
      {meta[0].shape, meta[0].dtype, 0}};
  if (measurement_mode) {
    output_attrs.push_back({meta[1].shape, meta[1].dtype, 1});
  }

  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe"sv, meta[0].dtype),
       std::move(inputs),
       output_attrs,
       params.get(),
       size});

  syn_out(0) = std::move(moe_result[0]);
  if (measurement_mode) {
    syn_out(1) = std::move(moe_result[1]);
  }
}

MixtureOfExpertsFwd::MixtureOfExpertsFwd(
    int device_id,
    c10::ScalarType scalar_type,
    bool recomp)
    : OpBackend(
          device_id,
          "moe_v2_fwd",
          scalar_type,
          std::vector<int>{0},
          {},
          {},
          false),
      recomp(recomp) {
  if (recomp) {
    SetOutputMetaFn(MixtureOfExpertsFwdRecompMeta);
  } else {
    SetOutputMetaFn(MixtureOfExpertsFwdMeta);
  }
}

void MixtureOfExpertsFwd::AddNode(sh::graph& graph, const at::Stack& stack) {
  const bool fused_weights = !stack.at(5).isTensorList();
  const size_t num_experts = stack.at(3).toTensorList().size();
  const size_t weights_per_expert = fused_weights ? 2 : 3;
  const size_t permute_weights_idx = fused_weights ? 5 : 6;
  const size_t output_number = recomp ? 1 : fused_weights ? 10 : 11;

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 3 + num_experts * weights_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }
  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, false, false);
  auto meta = OutputMeta(stack);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs = createOutputAttrs(meta);
  const std::string_view guid = recomp ? "moe_fwd"sv : "moe_v2_fwd"sv;
  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision(guid, meta[0].dtype),
       std::move(inputs),
       output_attrs,
       params.get(),
       size});
  for (size_t i = 0; i < output_number; ++i) {
    syn_out(i) = std::move(moe_result[i]);
  }
}

void HandleScaleScalar(
    habana::OpBackend* op,
    sh::graph& graph,
    const c10::IValue& scale,
    std::vector<sh::tensor>& scale_wrapper,
    std::vector<synTensor>& inputs,
    int insert_index = -1) {
  if (scale.isDouble()) {
    scale_wrapper.emplace_back(
        op->BuildConstantTensor(op, graph, scale.toDouble()));
    if (insert_index == -1) {
      inputs.push_back(scale_wrapper.back().get());
    } else {
      inputs.insert(inputs.begin() + insert_index, scale_wrapper.back().get());
    }
  } else if (scale.isDoubleList()) {
    auto scales_vector = scale.toDoubleVector();
    for (const auto& s : scales_vector) {
      scale_wrapper.emplace_back(op->BuildConstantTensor(op, graph, s));
      inputs.push_back(scale_wrapper.back().get());
    }
  }
}

void MixtureOfExpertsFp8::AddNode(sh::graph& graph, const at::Stack& stack) {
  const bool fused_weights = stack.size() == 13;
  const auto weights_and_scales_per_expert = fused_weights ? 5 : 7;
  const auto permuted_weights_idx = fused_weights ? 9 : 11;
  auto hidden_states = stack.at(0).toTensor();
  auto numExperts = stack.at(3).toTensorList().size();

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 4 + numExperts * weights_and_scales_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permuted_weights_idx, fused_weights, false, false);
  auto meta = MixtureOfExpertsFp8Meta(stack)[0];

  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe"sv, hidden_states.scalar_type()),
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});

  syn_out(0) = std::move(moe_result[0]);
}

void MixtureOfExpertsFp8Scalars::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  auto hidden_states = stack.at(0).toTensor();
  const bool fused_weights = !stack.at(5).isTensorList();
  auto num_experts = stack.at(3).toTensorList().size();
  auto weights_per_expert = fused_weights ? 2 : 3;
  auto permute_weights_idx = fused_weights ? 9 : 11;

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 3 + num_experts * weights_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }

  std::vector<sh::tensor> scale_wrapper;
  auto d_scale_hidden_states = stack.at(fused_weights ? 5 : 6);
  auto d_scale_intermediate_hidden_states = stack.at(fused_weights ? 6 : 7);
  auto d_scale_w1 = stack.at(fused_weights ? 7 : 8);
  auto d_scale_w2 = stack.at(fused_weights ? 8 : 9);

  HandleScaleScalar(this, graph, d_scale_hidden_states, scale_wrapper, inputs);
  HandleScaleScalar(
      this, graph, d_scale_intermediate_hidden_states, scale_wrapper, inputs);
  HandleScaleScalar(this, graph, d_scale_w1, scale_wrapper, inputs);
  HandleScaleScalar(this, graph, d_scale_w2, scale_wrapper, inputs);

  if (!fused_weights) {
    auto d_scale_w3 = stack.at(10);
    HandleScaleScalar(this, graph, d_scale_w3, scale_wrapper, inputs);
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, false, false);
  auto meta = MixtureOfExpertsFp8Meta(stack)[0];
  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe"sv, hidden_states.scalar_type()),
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});
  syn_out(0) = std::move(moe_result[0]);
}

void MixtureOfExpertsFp8Dynamic::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  const bool fused_weights = stack.size() == 12;
  const auto weights_and_scales_per_expert = fused_weights ? 4 : 6;
  const auto permuted_weights_idx = fused_weights ? 8 : 10;
  auto hidden_states = stack.at(0).toTensor();
  auto numExperts = stack.at(3).toTensorList().size();

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 4 + numExperts * weights_and_scales_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permuted_weights_idx, fused_weights, false, true);
  auto meta = MixtureOfExpertsFp8Meta(stack)[0];

  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe"sv, hidden_states.scalar_type()),
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});

  syn_out(0) = std::move(moe_result[0]);
}

void MixtureOfExpertsFp8ScalarsDynamic::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  auto hidden_states = stack.at(0).toTensor();
  const bool fused_weights = !stack.at(5).isTensorList();
  auto num_experts = stack.at(3).toTensorList().size();
  auto weights_per_expert = fused_weights ? 2 : 3;
  auto permute_weights_idx = fused_weights ? 8 : 10;

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 3 + num_experts * weights_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }

  std::vector<sh::tensor> scale_wrapper;
  auto d_scale_hidden_states = stack.at(fused_weights ? 5 : 6);
  auto d_scale_w1 = stack.at(fused_weights ? 6 : 7);
  auto d_scale_w2 = stack.at(fused_weights ? 7 : 8);

  HandleScaleScalar(this, graph, d_scale_hidden_states, scale_wrapper, inputs);
  HandleScaleScalar(this, graph, d_scale_w1, scale_wrapper, inputs);
  HandleScaleScalar(this, graph, d_scale_w2, scale_wrapper, inputs);

  if (!fused_weights) {
    auto d_scale_w3 = stack.at(9);
    HandleScaleScalar(this, graph, d_scale_w3, scale_wrapper, inputs);
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, false, true);
  auto meta = MixtureOfExpertsFp8Meta(stack)[0];
  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe"sv, hidden_states.scalar_type()),
       std::move(inputs),
       {{meta.shape, meta.dtype, 0}},
       params.get(),
       size});
  syn_out(0) = std::move(moe_result[0]);
}

MixtureOfExpertsBwd::MixtureOfExpertsBwd(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(device_id, "moe_v2_bwd", scalar_type, {0}, {}, {}, false) {
  SetOutputMetaFn(MixtureOfExpertsBwdMeta);
}

void MixtureOfExpertsBwd::AddNode(sh::graph& graph, const at::Stack& stack) {
  const bool fused_weights = !stack.at(12).isTensorList();
  const size_t first_weights_index = fused_weights ? 10 : 11;
  const size_t num_experts =
      stack.at(first_weights_index).toTensorList().size();
  const size_t weights_per_expert = fused_weights ? 2 : 3;
  const size_t permute_weights_idx = fused_weights ? 12 : 14;
  const size_t output_number = 2 + num_experts * weights_per_expert;
  const size_t non_list_tensors = fused_weights ? 10 : 11;

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < non_list_tensors + num_experts * weights_per_expert;
       i++) {
    inputs.push_back(syn_in(i));
  }

  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, false, false);
  auto meta = OutputMeta(stack);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs = createOutputAttrs(meta);
  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe_v2_bwd"sv, meta[0].dtype),
       std::move(inputs),
       output_attrs,
       params.get(),
       size});

  for (size_t i = 0; i < output_number; ++i) {
    syn_out(i) = std::move(moe_result[i]);
  }
}

MixtureOfExpertsRecompBwd::MixtureOfExpertsRecompBwd(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "moe_recomp_v2_bwd",
          scalar_type,
          {0},
          {},
          {},
          false) {
  SetOutputMetaFn(MixtureOfExpertsBwdMeta);
}

void MixtureOfExpertsRecompBwd::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  const bool fused_weights = !stack.at(6).isTensorList();
  const size_t num_experts = stack.at(4).toTensorList().size();
  const size_t weights_per_expert = fused_weights ? 2 : 3;
  const size_t permute_weights_idx = fused_weights ? 6 : 7;
  const size_t output_number = 2 + num_experts * weights_per_expert;

  std::vector<synTensor> inputs;
  for (size_t i = 0; i < 4 + num_experts * weights_per_expert; i++) {
    inputs.push_back(syn_in(i));
  }
  size_t size = 0;
  auto params = FillMixtureOfExpertsParams(
      stack, size, permute_weights_idx, fused_weights, false, false);
  auto meta = MixtureOfExpertsBwdMeta(stack);
  std::vector<NodeAttr::NodeOutputAttr> output_attrs = createOutputAttrs(meta);
  auto moe_result = OpBackend::BuildNode(
      this,
      graph,
      {get_guid_with_precision("moe_recomp_v2_bwd"sv, meta[0].dtype),
       std::move(inputs),
       output_attrs,
       params.get(),
       size});
  for (size_t i = 0; i < output_number; ++i) {
    syn_out(i) = std::move(moe_result[i]);
  }
}

} // namespace habana

static const auto& MixtureOfExpertsKernelRegistry =
    habana::KernelRegistry()
        .add("hpu::mixture_of_experts", KERNEL_FN_ARG(MixtureOfExperts, false))
        .add(
            "hpu::mixture_of_experts_fwd",
            KERNEL_FN_ARG(MixtureOfExpertsFwd, false))
        .add(
            "hpu::mixture_of_experts_recomp_fwd",
            KERNEL_FN_ARG(MixtureOfExpertsFwd, true))
        .add(
            "hpu::mixture_of_experts_fwd.fused_weights",
            KERNEL_FN_ARG(MixtureOfExpertsFwd, false))
        .add(
            "hpu::mixture_of_experts_recomp_fwd.fused_weights",
            KERNEL_FN_ARG(MixtureOfExpertsFwd, true))
        .add(
            "hpu::mixture_of_experts_bwd",
            KERNEL_FN_GLOBAL(habana::MixtureOfExpertsBwd))
        .add(
            "hpu::mixture_of_experts_bwd.fused_weights",
            KERNEL_FN_GLOBAL(habana::MixtureOfExpertsBwd))
        .add(
            "hpu::mixture_of_experts_recomp_bwd",
            KERNEL_FN_GLOBAL(habana::MixtureOfExpertsRecompBwd))
        .add(
            "hpu::mixture_of_experts_recomp_bwd.fused_weights",
            KERNEL_FN_GLOBAL(habana::MixtureOfExpertsRecompBwd))
        .add(
            "hpu::mixture_of_experts.fused_weights",
            KERNEL_FN_ARG(MixtureOfExperts, false))
        .add(
            "hpu::mixture_of_experts_fp8_measurement",
            KERNEL_FN_ARG(MixtureOfExperts, true))
        .add(
            "hpu::mixture_of_experts_fp8_measurement.fused_weights",
            KERNEL_FN_ARG(MixtureOfExperts, true));
