/**
 * Copyright (c) 2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#include <ATen/ATen.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "include/habanalabs/hpu_custom_op_pt2.h"
#include "include/habanalabs/triton_gaudi_launch.h"

namespace {

const std::string kFusedAddRmsNormSchema = "triton_gaudi::fused_add_rms_norm";
const std::string kDynamicQuantSchema = "triton_gaudi::dynamic_quant";
const std::string kSiluAndMulSchema = "triton_gaudi::silu_and_mul";
const std::string kGdnDecodePackedSchema =
    "triton_gaudi::gdn_decode_packed";
const std::string kGdnDecodeConvPackedSchema =
    "triton_gaudi::gdn_decode_conv_packed";
const std::string kGdnQkConvPackedSchema =
    "triton_gaudi::gdn_qk_conv_packed";
const std::string kGdnDecodeValueConvPackedSchema =
    "triton_gaudi::gdn_decode_value_conv_packed";

bool is_lower_hex_hash(const std::string& hash) {
  return hash.size() == habana::triton_gaudi::kArtifactHashChars &&
      std::all_of(hash.begin(), hash.end(), [](char value) {
        return (value >= '0' && value <= '9') ||
            (value >= 'a' && value <= 'f');
      });
}

void validate_inputs(
    const at::Tensor& hidden_states,
    const at::Tensor& residual,
    const at::Tensor& weight,
    std::int64_t block_size,
    std::int64_t n_cols) {
  TORCH_CHECK(
      hidden_states.scalar_type() == at::kBFloat16 &&
          residual.scalar_type() == at::kBFloat16 &&
          weight.scalar_type() == at::kBFloat16,
      "Triton Gaudi fused add+RMSNorm requires BF16 tensors");
  TORCH_CHECK(
      hidden_states.sizes() == residual.sizes() &&
          hidden_states.numel() > 0 && n_cols > 0 &&
          hidden_states.numel() % n_cols == 0 && weight.dim() == 1 &&
          weight.numel() == n_cols,
      "Triton Gaudi fused add+RMSNorm received incompatible tensor shapes");
  TORCH_CHECK(
      hidden_states.is_contiguous() && residual.is_contiguous() &&
          weight.is_contiguous(),
      "Triton Gaudi fused add+RMSNorm requires contiguous tensors");
  TORCH_CHECK(
      block_size > 0 && block_size <= 8192 &&
          (block_size & (block_size - 1)) == 0 && n_cols <= block_size &&
          (n_cols == 1 || n_cols > block_size / 2),
      "Triton Gaudi fused add+RMSNorm has invalid specialization metadata");
}

habana::PartialOutputMetaDataVector output_meta(const at::Stack& inputs) {
  const auto& hidden_states = inputs.at(0).toTensor();
  const auto& residual = inputs.at(1).toTensor();
  const auto& weight = inputs.at(2).toTensor();
  validate_inputs(
      hidden_states,
      residual,
      weight,
      inputs.at(4).toInt(),
      inputs.at(5).toInt());
  TORCH_CHECK(
      inputs.at(6).toInt() == hidden_states.numel() / inputs.at(5).toInt(),
      "Triton Gaudi fused add+RMSNorm row specialization is inconsistent");
  habana::PartialOutputMetaData output{
      hidden_states.scalar_type(), hidden_states.sizes().vec()};
  habana::PartialOutputMetaData residual_output{
      residual.scalar_type(), residual.sizes().vec()};
  return {output, residual_output};
}

std::shared_ptr<void> fill_params(const at::Stack& inputs, std::size_t& size) {
  const auto& hidden_states = inputs.at(0).toTensor();
  const auto& residual = inputs.at(1).toTensor();
  const auto& weight = inputs.at(2).toTensor();
  const std::string artifact_hash = inputs.at(3).toStringRef();
  const auto block_size = inputs.at(4).toInt();
  const auto n_cols = inputs.at(5).toInt();
  const auto rows = inputs.at(6).toInt();
  validate_inputs(hidden_states, residual, weight, block_size, n_cols);
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi fused add+RMSNorm received an invalid artifact hash");

  TORCH_CHECK(
      rows == hidden_states.numel() / n_cols && rows > 0 &&
          rows <= static_cast<std::int64_t>(
                      std::numeric_limits<std::uint32_t>::max()),
      "Triton Gaudi fused add+RMSNorm index space is out of range");

  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 3;
  params->output_count = 2;
  params->scalar_count = 1;
  params->index_space_rank = 1;
  params->block_size = static_cast<std::uint32_t>(block_size);
  params->logical_size = static_cast<std::uint32_t>(n_cols);
  params->tensor_dtype = 1U << 8; // tpc_lib_api::DATA_BF16
  params->kernel_kind = habana::triton_gaudi::KernelKind::FusedAddRmsNorm;
  params->grid[0] = static_cast<std::uint64_t>(rows);
  const float epsilon = static_cast<float>(inputs.at(7).toDouble());
  std::memcpy(&params->scalar_params[0], &epsilon, sizeof(epsilon));
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_fused_add_rms_norm() {
  habana::custom_op::registerUserCustomOp(
      kFusedAddRmsNormSchema,
      habana::triton_gaudi::kKernelGuid,
      output_meta,
      fill_params);
  return true;
}

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm(
    const at::Tensor& hidden_states,
    const at::Tensor& residual,
    const at::Tensor& weight,
    const std::string& artifact_hash,
    std::int64_t block_size,
    std::int64_t n_cols,
    std::int64_t rows,
    double epsilon) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::
      getUserCustomOpDescriptor(kFusedAddRmsNormSchema);
  std::vector<c10::IValue> inputs{
      hidden_states,
      residual,
      weight,
      artifact_hash,
      block_size,
      n_cols,
      rows,
      epsilon};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm_meta(
    const at::Tensor& hidden_states,
    const at::Tensor& residual,
    const at::Tensor&,
    const std::string&,
    std::int64_t,
    std::int64_t,
    std::int64_t,
    double) {
  return {at::empty_like(hidden_states), at::empty_like(residual)};
}

const bool kFusedAddRmsNormRegistered = register_fused_add_rms_norm();

void validate_dynamic_quant_input(
    const at::Tensor& input,
    std::int64_t block_size,
    std::int64_t n_cols,
    std::int64_t rows) {
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16,
      "Triton Gaudi dynamic quantization requires a BF16 input tensor");
  TORCH_CHECK(
      block_size > 0 && block_size <= 16384 &&
          (block_size & (block_size - 1)) == 0 && n_cols > 0 &&
          n_cols <= block_size && (n_cols == 1 || n_cols > block_size / 2),
      "Triton Gaudi dynamic quantization has invalid specialization metadata");
  TORCH_CHECK(
      rows > 0 &&
          rows <= static_cast<std::int64_t>(
                      std::numeric_limits<std::uint32_t>::max()) &&
          rows <= std::numeric_limits<std::int64_t>::max() / n_cols &&
          input.is_contiguous() && input.dim() == 1 && input.numel() > 0 &&
          input.numel() == n_cols * rows,
      "Triton Gaudi dynamic quantization received incompatible tensor storage");
}

habana::PartialOutputMetaDataVector dynamic_quant_output_meta(
    const at::Stack& inputs) {
  const auto& input = inputs.at(0).toTensor();
  const auto n_cols = inputs.at(3).toInt();
  const auto rows = inputs.at(4).toInt();
  validate_dynamic_quant_input(input, inputs.at(2).toInt(), n_cols, rows);
  habana::PartialOutputMetaData quantized{
      at::kFloat8_e4m3fn, input.sizes().vec()};
  habana::PartialOutputMetaData scale{at::kFloat, {rows}};
  return {quantized, scale};
}

std::shared_ptr<void> fill_dynamic_quant_params(
    const at::Stack& inputs,
    std::size_t& size) {
  const auto& input = inputs.at(0).toTensor();
  const std::string artifact_hash = inputs.at(1).toStringRef();
  const auto block_size = inputs.at(2).toInt();
  const auto n_cols = inputs.at(3).toInt();
  const auto rows = inputs.at(4).toInt();
  validate_dynamic_quant_input(input, block_size, n_cols, rows);
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi dynamic quantization received an invalid artifact hash");

  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 1;
  params->output_count = 2;
  params->scalar_count = 0;
  params->index_space_rank = 1;
  params->block_size = static_cast<std::uint32_t>(block_size);
  params->logical_size = static_cast<std::uint32_t>(n_cols);
  params->tensor_dtype = 1U << 5; // manifest primary/output dtype is E4M3
  params->kernel_kind = habana::triton_gaudi::KernelKind::DynamicQuant;
  params->grid[0] = static_cast<std::uint64_t>(rows);
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_dynamic_quant() {
  habana::custom_op::registerUserCustomOp(
      kDynamicQuantSchema,
      habana::triton_gaudi::kKernelGuid,
      dynamic_quant_output_meta,
      fill_dynamic_quant_params);
  return true;
}

std::tuple<at::Tensor, at::Tensor> dynamic_quant(
    const at::Tensor& input,
    const std::string& artifact_hash,
    std::int64_t block_size,
    std::int64_t n_cols,
    std::int64_t rows) {
  auto descriptor =
      habana::custom_op::UserCustomOpDescriptor::getUserCustomOpDescriptor(
          kDynamicQuantSchema);
  std::vector<c10::IValue> inputs{
      input, artifact_hash, block_size, n_cols, rows};
  auto outputs = descriptor.execute(inputs);
  return {outputs.at(0), outputs.at(1)};
}

std::tuple<at::Tensor, at::Tensor> dynamic_quant_meta(
    const at::Tensor& input,
    const std::string&,
    std::int64_t,
    std::int64_t,
    std::int64_t rows) {
  return {
      at::empty_like(input, input.options().dtype(at::kFloat8_e4m3fn)),
      at::empty({rows}, input.options().dtype(at::kFloat)),
  };
}

const bool kDynamicQuantRegistered = register_dynamic_quant();

void validate_silu_and_mul_input(
    const at::Tensor& input,
    std::int64_t block_size,
    std::int64_t n_cols,
    std::int64_t rows) {
  TORCH_CHECK(
      input.scalar_type() == at::kBFloat16,
      "Triton Gaudi SiLU-and-mul requires a BF16 input tensor");
  TORCH_CHECK(
      input.is_contiguous() && input.numel() > 0 && n_cols > 0 && rows > 0 &&
          input.dim() == 2 && input.size(0) == rows &&
          input.size(1) == 2 * n_cols && n_cols <= 65536 &&
          rows <= static_cast<std::int64_t>(
                      std::numeric_limits<std::uint32_t>::max()) &&
          input.numel() == 2 * rows * n_cols,
      "Triton Gaudi SiLU-and-mul received incompatible tensor storage");
  TORCH_CHECK(
      block_size >= 128 && block_size <= 1024 &&
          (block_size & (block_size - 1)) == 0,
      "Triton Gaudi SiLU-and-mul has invalid specialization metadata");
}

habana::PartialOutputMetaDataVector silu_and_mul_output_meta(
    const at::Stack& inputs) {
  const auto& input = inputs.at(0).toTensor();
  const auto n_cols = inputs.at(3).toInt();
  const auto rows = inputs.at(4).toInt();
  validate_silu_and_mul_input(
      input, inputs.at(2).toInt(), n_cols, rows);
  habana::PartialOutputMetaData output{
      input.scalar_type(), {rows, n_cols}};
  return {output};
}

std::shared_ptr<void> fill_silu_and_mul_params(
    const at::Stack& inputs,
    std::size_t& size) {
  const auto& input = inputs.at(0).toTensor();
  const std::string artifact_hash = inputs.at(1).toStringRef();
  const auto block_size = inputs.at(2).toInt();
  const auto n_cols = inputs.at(3).toInt();
  const auto rows = inputs.at(4).toInt();
  validate_silu_and_mul_input(input, block_size, n_cols, rows);
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi SiLU-and-mul received an invalid artifact hash");

  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 1;
  params->output_count = 1;
  params->scalar_count = 0;
  params->index_space_rank = 2;
  params->block_size = static_cast<std::uint32_t>(block_size);
  params->logical_size = static_cast<std::uint32_t>(n_cols);
  params->tensor_dtype = 1U << 8; // tpc_lib_api::DATA_BF16
  params->kernel_kind = habana::triton_gaudi::KernelKind::SiluAndMul;
  params->grid[0] = static_cast<std::uint64_t>(
      (n_cols + block_size - 1) / block_size);
  params->grid[1] = static_cast<std::uint64_t>(rows);
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_silu_and_mul() {
  habana::custom_op::registerUserCustomOp(
      kSiluAndMulSchema,
      habana::triton_gaudi::kKernelGuid,
      silu_and_mul_output_meta,
      fill_silu_and_mul_params);
  return true;
}

at::Tensor silu_and_mul(
    const at::Tensor& input,
    const std::string& artifact_hash,
    std::int64_t block_size,
    std::int64_t n_cols,
    std::int64_t rows) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::
      getUserCustomOpDescriptor(kSiluAndMulSchema);
  std::vector<c10::IValue> inputs{
      input, artifact_hash, block_size, n_cols, rows};
  return descriptor.execute(inputs).at(0);
}

at::Tensor silu_and_mul_meta(
    const at::Tensor& input,
    const std::string&,
    std::int64_t,
    std::int64_t n_cols,
    std::int64_t rows) {
  return at::empty({rows, n_cols}, input.options());
}

const bool kSiluAndMulRegistered = register_silu_and_mul();

void validate_gdn_decode_packed_inputs(
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& gate_a,
    const at::Tensor& gate_b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& state_indices,
    std::int64_t value_tile) {
  TORCH_CHECK(
      state_cache.scalar_type() == at::kFloat &&
          packed_qkv.scalar_type() == at::kBFloat16 &&
          gate_a.scalar_type() == at::kBFloat16 &&
          gate_b.scalar_type() == at::kBFloat16 &&
          a_log.scalar_type() == at::kFloat &&
          dt_bias.scalar_type() == at::kFloat &&
          state_indices.scalar_type() == at::kInt,
      "Triton Gaudi packed GDN decode requires the canonical mixed dtypes");
  TORCH_CHECK(
      state_cache.dim() == 4 && state_cache.size(0) > 0 &&
          state_cache.size(1) == 48 && state_cache.size(2) == 128 &&
          state_cache.size(3) == 128 && packed_qkv.dim() == 2 &&
          packed_qkv.size(0) > 0 && packed_qkv.size(1) == 10240,
      "Triton Gaudi packed GDN decode requires Qwen3.5 state and packed QKV shapes");
  const auto batch = packed_qkv.size(0);
  TORCH_CHECK(
      gate_a.sizes() == at::IntArrayRef({batch, 48}) &&
          gate_b.sizes() == gate_a.sizes() &&
          a_log.sizes() == at::IntArrayRef({48}) &&
          dt_bias.sizes() == at::IntArrayRef({48}) &&
          state_indices.sizes() == at::IntArrayRef({batch}),
      "Triton Gaudi packed GDN decode received incompatible gate or index shapes");
  TORCH_CHECK(
      state_cache.is_contiguous() && packed_qkv.is_contiguous() &&
          gate_a.is_contiguous() && gate_b.is_contiguous() &&
          a_log.is_contiguous() && dt_bias.is_contiguous() &&
          state_indices.is_contiguous(),
      "Triton Gaudi packed GDN decode requires contiguous tensors");
  TORCH_CHECK(
      value_tile == 16 || value_tile == 32 || value_tile == 64 ||
          value_tile == 128,
      "Triton Gaudi packed GDN decode VALUE_TILE must be 16, 32, 64, or 128");
  TORCH_CHECK(
      state_cache.size(0) <=
          static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()) &&
          batch <=
              static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max()),
      "Triton Gaudi packed GDN decode geometry exceeds the launch ABI");
}

habana::PartialOutputMetaDataVector gdn_decode_packed_output_meta(
    const at::Stack& inputs) {
  const auto& state_cache = inputs.at(0).toTensor();
  const auto& packed_qkv = inputs.at(1).toTensor();
  validate_gdn_decode_packed_inputs(
      state_cache,
      packed_qkv,
      inputs.at(2).toTensor(),
      inputs.at(3).toTensor(),
      inputs.at(4).toTensor(),
      inputs.at(5).toTensor(),
      inputs.at(6).toTensor(),
      inputs.at(8).toInt());
  return {{at::kBFloat16, {packed_qkv.size(0), 48, 128}}};
}

std::shared_ptr<void> fill_gdn_decode_packed_params(
    const at::Stack& inputs,
    std::size_t& size) {
  const auto& state_cache = inputs.at(0).toTensor();
  const auto& packed_qkv = inputs.at(1).toTensor();
  const std::string artifact_hash = inputs.at(7).toStringRef();
  const auto value_tile = inputs.at(8).toInt();
  validate_gdn_decode_packed_inputs(
      state_cache,
      packed_qkv,
      inputs.at(2).toTensor(),
      inputs.at(3).toTensor(),
      inputs.at(4).toTensor(),
      inputs.at(5).toTensor(),
      inputs.at(6).toTensor(),
      value_tile);
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi packed GDN decode received an invalid artifact hash");

  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 7;
  params->output_count = 1;
  params->scalar_count = 1;
  params->index_space_rank = 3;
  params->block_size = static_cast<std::uint32_t>(value_tile);
  params->logical_size = 128;
  params->tensor_dtype = 1U << 8; // primary/output dtype is BF16
  params->kernel_kind =
      habana::triton_gaudi::KernelKind::GdnDecodePacked;
  params->grid[0] = static_cast<std::uint64_t>(128 / value_tile);
  params->grid[1] = 48;
  params->grid[2] = static_cast<std::uint64_t>(packed_qkv.size(0));
  params->scalar_params[0] =
      static_cast<std::uint32_t>(state_cache.size(0));
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_gdn_decode_packed() {
  habana::custom_op::registerUserCustomOp(
      kGdnDecodePackedSchema,
      habana::triton_gaudi::kKernelGuid,
      gdn_decode_packed_output_meta,
      fill_gdn_decode_packed_params);
  return true;
}

at::Tensor gdn_decode_packed(
    at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& gate_a,
    const at::Tensor& gate_b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& state_indices,
    const std::string& artifact_hash,
    std::int64_t value_tile) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::
      getUserCustomOpDescriptor(kGdnDecodePackedSchema);
  std::vector<c10::IValue> inputs{
      state_cache,
      packed_qkv,
      gate_a,
      gate_b,
      a_log,
      dt_bias,
      state_indices,
      artifact_hash,
      value_tile};
  return descriptor.execute(inputs).at(0);
}

at::Tensor gdn_decode_packed_meta(
    at::Tensor&,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::string&,
    std::int64_t) {
  return at::empty(
      {packed_qkv.size(0), 48, 128},
      packed_qkv.options().dtype(at::kBFloat16));
}

const bool kGdnDecodePackedRegistered = register_gdn_decode_packed();

void validate_gdn_decode_conv_packed_inputs(
    const at::Tensor& conv_state,
    const at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& gate_a,
    const at::Tensor& gate_b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& state_indices,
    const at::Tensor& conv_weight_t) {
  TORCH_CHECK(
      conv_state.scalar_type() == at::kBFloat16 &&
          state_cache.scalar_type() == at::kFloat &&
          packed_qkv.scalar_type() == at::kBFloat16 &&
          gate_a.scalar_type() == at::kBFloat16 &&
          gate_b.scalar_type() == at::kBFloat16 &&
          a_log.scalar_type() == at::kFloat &&
          dt_bias.scalar_type() == at::kFloat &&
          state_indices.scalar_type() == at::kInt &&
          conv_weight_t.scalar_type() == at::kBFloat16,
      "Triton Gaudi fused conv+GDN requires the canonical mixed dtypes");
  TORCH_CHECK(
      conv_state.dim() == 3 && conv_state.size(0) > 0 &&
          conv_state.size(1) == 3 && conv_state.size(2) == 10240 &&
          state_cache.dim() == 4 && state_cache.size(0) > 0 &&
          state_cache.size(1) == 48 && state_cache.size(2) == 128 &&
          state_cache.size(3) == 128 && packed_qkv.dim() == 2 &&
          packed_qkv.size(0) > 0 && packed_qkv.size(1) == 10240,
      "Triton Gaudi fused conv+GDN requires Qwen3.5 cache and packed QKV shapes");
  const auto batch = packed_qkv.size(0);
  TORCH_CHECK(
      gate_a.sizes() == at::IntArrayRef({batch, 48}) &&
          gate_b.sizes() == gate_a.sizes() &&
          a_log.sizes() == at::IntArrayRef({48}) &&
          dt_bias.sizes() == at::IntArrayRef({48}) &&
          state_indices.sizes() == at::IntArrayRef({batch}) &&
          conv_weight_t.sizes() == at::IntArrayRef({4, 10240}),
      "Triton Gaudi fused conv+GDN received incompatible weights, gates, or indices");
  TORCH_CHECK(
      conv_state.is_contiguous() && state_cache.is_contiguous() &&
          packed_qkv.is_contiguous() && gate_a.is_contiguous() &&
          gate_b.is_contiguous() && a_log.is_contiguous() &&
          dt_bias.is_contiguous() && state_indices.is_contiguous() &&
          conv_weight_t.is_contiguous(),
      "Triton Gaudi fused conv+GDN requires contiguous tensors");
  TORCH_CHECK(
      conv_state.size(0) <=
              static_cast<std::int64_t>(
                  std::numeric_limits<std::uint32_t>::max()) &&
          state_cache.size(0) <=
              static_cast<std::int64_t>(
                  std::numeric_limits<std::uint32_t>::max()) &&
          batch <=
              static_cast<std::int64_t>(
                  std::numeric_limits<std::uint32_t>::max()),
      "Triton Gaudi fused conv+GDN geometry exceeds the launch ABI");
}

habana::PartialOutputMetaDataVector gdn_decode_conv_packed_output_meta(
    const at::Stack& inputs) {
  const auto& packed_qkv = inputs.at(2).toTensor();
  validate_gdn_decode_conv_packed_inputs(
      inputs.at(0).toTensor(),
      inputs.at(1).toTensor(),
      packed_qkv,
      inputs.at(3).toTensor(),
      inputs.at(4).toTensor(),
      inputs.at(5).toTensor(),
      inputs.at(6).toTensor(),
      inputs.at(7).toTensor(),
      inputs.at(8).toTensor());
  return {{at::kBFloat16, {packed_qkv.size(0), 48, 128}}};
}

std::shared_ptr<void> fill_gdn_decode_conv_packed_params(
    const at::Stack& inputs,
    std::size_t& size) {
  const auto& conv_state = inputs.at(0).toTensor();
  const auto& state_cache = inputs.at(1).toTensor();
  const auto& packed_qkv = inputs.at(2).toTensor();
  const std::string artifact_hash = inputs.at(9).toStringRef();
  validate_gdn_decode_conv_packed_inputs(
      conv_state,
      state_cache,
      packed_qkv,
      inputs.at(3).toTensor(),
      inputs.at(4).toTensor(),
      inputs.at(5).toTensor(),
      inputs.at(6).toTensor(),
      inputs.at(7).toTensor(),
      inputs.at(8).toTensor());
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi fused conv+GDN received an invalid artifact hash");

  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 9;
  params->output_count = 1;
  params->scalar_count = 2;
  params->index_space_rank = 2;
  params->block_size = 128;
  params->logical_size = 128;
  params->tensor_dtype = 1U << 8; // primary/output dtype is BF16
  params->kernel_kind =
      habana::triton_gaudi::KernelKind::GdnDecodeConvPacked;
  params->grid[0] = 16;
  params->grid[1] = static_cast<std::uint64_t>(packed_qkv.size(0));
  params->scalar_params[0] =
      static_cast<std::uint32_t>(conv_state.size(0));
  params->scalar_params[1] =
      static_cast<std::uint32_t>(state_cache.size(0));
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_gdn_decode_conv_packed() {
  habana::custom_op::registerUserCustomOp(
      kGdnDecodeConvPackedSchema,
      habana::triton_gaudi::kKernelGuid,
      gdn_decode_conv_packed_output_meta,
      fill_gdn_decode_conv_packed_params);
  return true;
}

at::Tensor gdn_decode_conv_packed(
    at::Tensor& conv_state,
    at::Tensor& state_cache,
    const at::Tensor& packed_qkv,
    const at::Tensor& gate_a,
    const at::Tensor& gate_b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& state_indices,
    const at::Tensor& conv_weight_t,
    const std::string& artifact_hash) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::
      getUserCustomOpDescriptor(kGdnDecodeConvPackedSchema);
  std::vector<c10::IValue> inputs{
      conv_state,
      state_cache,
      packed_qkv,
      gate_a,
      gate_b,
      a_log,
      dt_bias,
      state_indices,
      conv_weight_t,
      artifact_hash};
  return descriptor.execute(inputs).at(0);
}

at::Tensor gdn_decode_conv_packed_meta(
    at::Tensor&,
    at::Tensor&,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::string&) {
  return at::empty(
      {packed_qkv.size(0), 48, 128},
      packed_qkv.options().dtype(at::kBFloat16));
}

const bool kGdnDecodeConvPackedRegistered =
    register_gdn_decode_conv_packed();

void validate_gdn_qk_conv_packed_inputs(
    const at::Tensor& conv_state,
    const at::Tensor& packed_qkv,
    const at::Tensor& state_indices,
    const at::Tensor& conv_weight_t) {
  TORCH_CHECK(
      conv_state.scalar_type() == at::kBFloat16 &&
          packed_qkv.scalar_type() == at::kBFloat16 &&
          state_indices.scalar_type() == at::kInt &&
          conv_weight_t.scalar_type() == at::kBFloat16,
      "Triton Gaudi packed Q/K convolution requires BF16 data and i32 indices");
  TORCH_CHECK(
      conv_state.dim() == 3 && conv_state.size(0) > 0 &&
          conv_state.size(1) == 3 && conv_state.size(2) == 10240 &&
          packed_qkv.dim() == 2 && packed_qkv.size(0) > 0 &&
          packed_qkv.size(1) == 10240 &&
          state_indices.sizes() == at::IntArrayRef({packed_qkv.size(0)}) &&
          conv_weight_t.sizes() == at::IntArrayRef({4, 10240}),
      "Triton Gaudi packed Q/K convolution received incompatible shapes");
  TORCH_CHECK(
      conv_state.is_contiguous() && packed_qkv.is_contiguous() &&
          state_indices.is_contiguous() && conv_weight_t.is_contiguous(),
      "Triton Gaudi packed Q/K convolution requires contiguous tensors");
  TORCH_CHECK(
      conv_state.size(0) <=
              static_cast<std::int64_t>(
                  std::numeric_limits<std::uint32_t>::max()) &&
          packed_qkv.size(0) <=
              static_cast<std::int64_t>(
                  std::numeric_limits<std::uint32_t>::max()),
      "Triton Gaudi packed Q/K convolution geometry exceeds the launch ABI");
}

habana::PartialOutputMetaDataVector gdn_qk_conv_packed_output_meta(
    const at::Stack& inputs) {
  const auto& packed_qkv = inputs.at(1).toTensor();
  validate_gdn_qk_conv_packed_inputs(
      inputs.at(0).toTensor(),
      packed_qkv,
      inputs.at(2).toTensor(),
      inputs.at(3).toTensor());
  return {{at::kBFloat16, {packed_qkv.size(0), 4096}}};
}

std::shared_ptr<void> fill_gdn_qk_conv_packed_params(
    const at::Stack& inputs,
    std::size_t& size) {
  const auto& conv_state = inputs.at(0).toTensor();
  const auto& packed_qkv = inputs.at(1).toTensor();
  const std::string artifact_hash = inputs.at(4).toStringRef();
  validate_gdn_qk_conv_packed_inputs(
      conv_state,
      packed_qkv,
      inputs.at(2).toTensor(),
      inputs.at(3).toTensor());
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi packed Q/K convolution received an invalid artifact hash");
  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 4;
  params->output_count = 1;
  params->scalar_count = 1;
  params->index_space_rank = 2;
  params->block_size = 128;
  params->logical_size = 4096;
  params->tensor_dtype = 1U << 8;
  params->kernel_kind = habana::triton_gaudi::KernelKind::GdnQkConvPacked;
  params->grid[0] = 32;
  params->grid[1] = static_cast<std::uint64_t>(packed_qkv.size(0));
  params->scalar_params[0] =
      static_cast<std::uint32_t>(conv_state.size(0));
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_gdn_qk_conv_packed() {
  habana::custom_op::registerUserCustomOp(
      kGdnQkConvPackedSchema,
      habana::triton_gaudi::kKernelGuid,
      gdn_qk_conv_packed_output_meta,
      fill_gdn_qk_conv_packed_params);
  return true;
}

at::Tensor gdn_qk_conv_packed(
    at::Tensor& conv_state,
    const at::Tensor& packed_qkv,
    const at::Tensor& state_indices,
    const at::Tensor& conv_weight_t,
    const std::string& artifact_hash) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::
      getUserCustomOpDescriptor(kGdnQkConvPackedSchema);
  std::vector<c10::IValue> inputs{
      conv_state,
      packed_qkv,
      state_indices,
      conv_weight_t,
      artifact_hash};
  return descriptor.execute(inputs).at(0);
}

at::Tensor gdn_qk_conv_packed_meta(
    at::Tensor&,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const std::string&) {
  return at::empty(
      {packed_qkv.size(0), 4096},
      packed_qkv.options().dtype(at::kBFloat16));
}

const bool kGdnQkConvPackedRegistered = register_gdn_qk_conv_packed();

void validate_gdn_decode_value_conv_packed_inputs(
    const at::Tensor& conv_state,
    const at::Tensor& state_cache,
    const at::Tensor& qk_conv,
    const at::Tensor& packed_qkv,
    const at::Tensor& gate_a,
    const at::Tensor& gate_b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& state_indices,
    const at::Tensor& conv_weight_t,
    std::int64_t value_tile) {
  validate_gdn_decode_conv_packed_inputs(
      conv_state,
      state_cache,
      packed_qkv,
      gate_a,
      gate_b,
      a_log,
      dt_bias,
      state_indices,
      conv_weight_t);
  TORCH_CHECK(
      qk_conv.scalar_type() == at::kBFloat16 && qk_conv.is_contiguous() &&
          qk_conv.sizes() == at::IntArrayRef({packed_qkv.size(0), 4096}),
      "Triton Gaudi fused value-conv + GDN requires packed BF16 Q/K input");
  TORCH_CHECK(
      value_tile == 16 || value_tile == 32 || value_tile == 64 ||
          value_tile == 128,
      "Triton Gaudi fused value-conv + GDN has invalid VALUE_TILE");
}

habana::PartialOutputMetaDataVector
gdn_decode_value_conv_packed_output_meta(const at::Stack& inputs) {
  const auto& packed_qkv = inputs.at(3).toTensor();
  validate_gdn_decode_value_conv_packed_inputs(
      inputs.at(0).toTensor(),
      inputs.at(1).toTensor(),
      inputs.at(2).toTensor(),
      packed_qkv,
      inputs.at(4).toTensor(),
      inputs.at(5).toTensor(),
      inputs.at(6).toTensor(),
      inputs.at(7).toTensor(),
      inputs.at(8).toTensor(),
      inputs.at(9).toTensor(),
      inputs.at(11).toInt());
  return {{at::kBFloat16, {packed_qkv.size(0), 48, 128}}};
}

std::shared_ptr<void> fill_gdn_decode_value_conv_packed_params(
    const at::Stack& inputs,
    std::size_t& size) {
  const auto& conv_state = inputs.at(0).toTensor();
  const auto& state_cache = inputs.at(1).toTensor();
  const auto& packed_qkv = inputs.at(3).toTensor();
  const std::string artifact_hash = inputs.at(10).toStringRef();
  const auto value_tile = inputs.at(11).toInt();
  validate_gdn_decode_value_conv_packed_inputs(
      conv_state,
      state_cache,
      inputs.at(2).toTensor(),
      packed_qkv,
      inputs.at(4).toTensor(),
      inputs.at(5).toTensor(),
      inputs.at(6).toTensor(),
      inputs.at(7).toTensor(),
      inputs.at(8).toTensor(),
      inputs.at(9).toTensor(),
      value_tile);
  TORCH_CHECK(
      is_lower_hex_hash(artifact_hash),
      "Triton Gaudi fused value-conv + GDN received an invalid artifact hash");
  HPU_PARAMS_STUB(habana::triton_gaudi::LaunchParamsV1);
  params->input_count = 10;
  params->output_count = 1;
  params->scalar_count = 2;
  params->index_space_rank = 3;
  params->block_size = static_cast<std::uint32_t>(value_tile);
  params->logical_size = 128;
  params->tensor_dtype = 1U << 8;
  params->kernel_kind =
      habana::triton_gaudi::KernelKind::GdnDecodeValueConvPacked;
  params->grid[0] = static_cast<std::uint64_t>(128 / value_tile);
  params->grid[1] = 48;
  params->grid[2] = static_cast<std::uint64_t>(packed_qkv.size(0));
  params->scalar_params[0] =
      static_cast<std::uint32_t>(conv_state.size(0));
  params->scalar_params[1] =
      static_cast<std::uint32_t>(state_cache.size(0));
  std::copy(
      artifact_hash.begin(),
      artifact_hash.end(),
      params->artifact_hash.begin());
  return params;
}

bool register_gdn_decode_value_conv_packed() {
  habana::custom_op::registerUserCustomOp(
      kGdnDecodeValueConvPackedSchema,
      habana::triton_gaudi::kKernelGuid,
      gdn_decode_value_conv_packed_output_meta,
      fill_gdn_decode_value_conv_packed_params);
  return true;
}

at::Tensor gdn_decode_value_conv_packed(
    at::Tensor& conv_state,
    at::Tensor& state_cache,
    const at::Tensor& qk_conv,
    const at::Tensor& packed_qkv,
    const at::Tensor& gate_a,
    const at::Tensor& gate_b,
    const at::Tensor& a_log,
    const at::Tensor& dt_bias,
    const at::Tensor& state_indices,
    const at::Tensor& conv_weight_t,
    const std::string& artifact_hash,
    std::int64_t value_tile) {
  auto descriptor = habana::custom_op::UserCustomOpDescriptor::
      getUserCustomOpDescriptor(kGdnDecodeValueConvPackedSchema);
  std::vector<c10::IValue> inputs{
      conv_state,
      state_cache,
      qk_conv,
      packed_qkv,
      gate_a,
      gate_b,
      a_log,
      dt_bias,
      state_indices,
      conv_weight_t,
      artifact_hash,
      value_tile};
  return descriptor.execute(inputs).at(0);
}

at::Tensor gdn_decode_value_conv_packed_meta(
    at::Tensor&,
    at::Tensor&,
    const at::Tensor&,
    const at::Tensor& packed_qkv,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const at::Tensor&,
    const std::string&,
    std::int64_t) {
  return at::empty(
      {packed_qkv.size(0), 48, 128},
      packed_qkv.options().dtype(at::kBFloat16));
}

const bool kGdnDecodeValueConvPackedRegistered =
    register_gdn_decode_value_conv_packed();

} // namespace

TORCH_LIBRARY_FRAGMENT(triton_gaudi, module) {
  module.def(
      "fused_add_rms_norm(Tensor hidden_states, Tensor residual, Tensor weight, "
      "str artifact_hash, int block_size, int n_cols, int rows, float epsilon) "
      "-> (Tensor, Tensor)");
  module.def(
      "dynamic_quant(Tensor input, str artifact_hash, int block_size, "
      "int n_cols, int rows) -> (Tensor, Tensor)");
  module.def(
      "silu_and_mul(Tensor input, str artifact_hash, int block_size, "
      "int n_cols, int rows) -> Tensor");
  module.def(
      "gdn_decode_packed(Tensor(a!) state_cache, Tensor packed_qkv, "
      "Tensor gate_a, Tensor gate_b, Tensor a_log, Tensor dt_bias, "
      "Tensor state_indices, str artifact_hash, int value_tile) -> Tensor");
  module.def(
      "gdn_decode_conv_packed(Tensor(a!) conv_state, "
      "Tensor(b!) state_cache, Tensor packed_qkv, Tensor gate_a, "
      "Tensor gate_b, Tensor a_log, Tensor dt_bias, Tensor state_indices, "
      "Tensor conv_weight_t, str artifact_hash) -> Tensor");
  module.def(
      "gdn_qk_conv_packed(Tensor(a!) conv_state, Tensor packed_qkv, "
      "Tensor state_indices, Tensor conv_weight_t, str artifact_hash) "
      "-> Tensor");
  module.def(
      "gdn_decode_value_conv_packed(Tensor(a!) conv_state, "
      "Tensor(b!) state_cache, Tensor qk_conv, Tensor packed_qkv, "
      "Tensor gate_a, Tensor gate_b, Tensor a_log, Tensor dt_bias, "
      "Tensor state_indices, Tensor conv_weight_t, str artifact_hash, "
      "int value_tile) -> Tensor");
}

TORCH_LIBRARY_IMPL(triton_gaudi, HPU, module) {
  module.impl("fused_add_rms_norm", fused_add_rms_norm);
  module.impl("dynamic_quant", dynamic_quant);
  module.impl("silu_and_mul", silu_and_mul);
  module.impl("gdn_decode_packed", gdn_decode_packed);
  module.impl("gdn_decode_conv_packed", gdn_decode_conv_packed);
  module.impl("gdn_qk_conv_packed", gdn_qk_conv_packed);
  module.impl(
      "gdn_decode_value_conv_packed",
      gdn_decode_value_conv_packed);
}

TORCH_LIBRARY_IMPL(triton_gaudi, Meta, module) {
  module.impl("fused_add_rms_norm", fused_add_rms_norm_meta);
  module.impl("dynamic_quant", dynamic_quant_meta);
  module.impl("silu_and_mul", silu_and_mul_meta);
  module.impl("gdn_decode_packed", gdn_decode_packed_meta);
  module.impl("gdn_decode_conv_packed", gdn_decode_conv_packed_meta);
  module.impl("gdn_qk_conv_packed", gdn_qk_conv_packed_meta);
  module.impl(
      "gdn_decode_value_conv_packed",
      gdn_decode_value_conv_packed_meta);
}
