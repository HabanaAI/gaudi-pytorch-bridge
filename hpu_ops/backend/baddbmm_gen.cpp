/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/baddbmm.h"

namespace {
std::vector<synapse_helpers::tensor> ComputeGEMM(
    habana::OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const habana::OutputMetaData& meta,
    c10::optional<int> final_idx = c10::nullopt) {
  habana::NodeAttr::NodeOutputAttr gemm_node_output_attr = {
      meta.shape, meta.dtype};
  gemm_node_output_attr.final_result_index = final_idx;
  synGEMMParams matmul_params{};
  std::vector<synapse_helpers::tensor> gemm_out = habana::OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       std::move(input_tensor),
       {gemm_node_output_attr},
       &matmul_params,
       sizeof(matmul_params)});
  return gemm_out;
}
} // namespace

namespace habana {

OutputMetaDataVector BaddbmmMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto batch1 = stack_tensor(stack, 1);
  auto batch2 = stack_tensor(stack, 2);
  const auto batch1_sizes = batch1.sizes();
  const auto batch2_sizes = batch2.sizes();
  TORCH_CHECK(batch1.dim() == 3, "batch1 must be a 3D tensor");
  TORCH_CHECK(batch2.dim() == 3, "batch2 must be a 3D tensor");
  int64_t bs = batch1_sizes[0];
  int64_t contraction_size = batch1_sizes[2];
  int64_t res_rows = batch1_sizes[1];
  int64_t res_cols = batch2_sizes[2];
  OutputMetaData meta;
  meta.shape = {bs, res_rows, res_cols};
  meta.dtype = self.scalar_type();

  TORCH_CHECK(
      batch2_sizes[0] == bs && batch2_sizes[1] == contraction_size,
      "Expected size for first two dimensions of batch2 tensor to be: [",
      bs,
      ", ",
      contraction_size,
      "] but got: [",
      batch2_sizes[0],
      ", ",
      batch2_sizes[1],
      "].");

  return {meta};
}

static std::vector<synapse_helpers::tensor> ComputeBetaSide(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const OutputMetaData& meta,
    const float beta_val,
    c10::optional<int> final_idx = c10::nullopt) {
  synapse_helpers::tensor beta_tensor =
      OpBackend::BuildConstant(op, graph, beta_val, meta.dtype, meta.shape);

  std::vector<synTensor> node_inputs{input_tensor.at(0), beta_tensor.get()};
  std::vector<synapse_helpers::tensor> beta_side_out = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult", meta.dtype),
       std::move(node_inputs),
       {{meta.shape, meta.dtype, final_idx}}});

  return beta_side_out;
}

static std::vector<synapse_helpers::tensor> ComputeAlphaSide(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const OutputMetaData& meta,
    const float alpha_val,
    c10::optional<int> final_idx = c10::nullopt) {
  c10::optional<int> is_gemm_final_node = c10::nullopt;
  if (alpha_val == 1.0) {
    is_gemm_final_node = final_idx;
  }

  std::vector<synapse_helpers::tensor> gemm_out =
      ComputeGEMM(op, graph, input_tensor, meta, is_gemm_final_node);

  if (alpha_val == 1.0) {
    return gemm_out;
  } else {
    auto alpha_tensor =
        OpBackend::BuildConstant(op, graph, alpha_val, meta.dtype, meta.shape);
    std::vector<synTensor> mul_node_inputs{
        gemm_out[0].get(), alpha_tensor.get()};
    std::vector<synapse_helpers::tensor> alpha_mul_out = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("mult", meta.dtype),
         std::move(mul_node_inputs),
         {{meta.shape, meta.dtype, final_idx}}});
    return alpha_mul_out;
  }
}

static std::vector<synapse_helpers::tensor> BaddbMMCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> input_tensor,
    const OutputMetaData& meta) {
  std::vector<synapse_helpers::tensor> baddbmm_out;

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  if (alpha_val == 0.0 && beta_val == 0.0) {
    baddbmm_out.emplace_back(
        OpBackend::BuildConstant(op, graph, 0.0, meta.dtype, meta.shape, 0));
  } else if (alpha_val == 0.0 && beta_val != 0.0) {
    baddbmm_out =
        ComputeBetaSide(op, graph, {input_tensor.at(0)}, meta, beta_val, 0);
  } else if (alpha_val != 0.0 && beta_val == 0.0) {
    baddbmm_out = ComputeAlphaSide(
        op,
        graph,
        {input_tensor.at(1), input_tensor.at(2)},
        meta,
        alpha_val,
        0);
  } else {
    auto beta_out =
        ComputeBetaSide(op, graph, {input_tensor.at(0)}, meta, beta_val);
    auto alpha_out = ComputeAlphaSide(
        op, graph, {input_tensor.at(1), input_tensor.at(2)}, meta, alpha_val);
    std::vector<synTensor> add_node_inputs{
        beta_out[0].get(), alpha_out[0].get()};
    baddbmm_out = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("add", meta.dtype),
         std::move(add_node_inputs),
         {{meta.shape, meta.dtype, 0}}});
  }
  return baddbmm_out;
}

static SharedMetaData BetaSharedMeta(
    int input_rank,
    at::ScalarType input_dtype) {
  SharedMetaData mul{"mult_fwd"};
  mul.inputs_data = {{input_rank, input_dtype}, {3, input_dtype}};
  mul.outputs_data = {{3, input_dtype}};
  return mul;
}

static SharedMetaDataVector AlphaSharedMeta(
    at::ScalarType batch1_dtype,
    at::ScalarType batch2_dtype,
    float alpha) {
  SharedMetaTensor common_data = {3, batch1_dtype};
  SharedMetaData bmm{"batch_gemm"};
  bmm.inputs_data = {common_data, {3, batch2_dtype}};
  bmm.outputs_data = {common_data};

  if (alpha != 1.0) {
    SharedMetaData mul{"mult_fwd"};
    mul.inputs_data = {2, common_data};
    mul.outputs_data = {common_data};
    return {bmm, mul};
  }

  return {bmm};
}

SharedMetaDataVector BAddBMMSharedMeta(const at::Stack& stack) {
  const float beta = stack.at(3).toScalar().toFloat();
  const float alpha = stack.at(4).toScalar().toFloat();

  if (alpha == 0.0 and beta == 0.0) {
    return {};
  }

  auto input = stack_tensor(stack, 0);
  auto input_dtype = input.scalar_type();
  auto input_rank = input.dim();

  if (alpha == 0.0) {
    return {BetaSharedMeta(input_rank, input_dtype)};
  }

  auto batch1 = stack_tensor(stack, 1);
  auto batch1_dtype = batch1.scalar_type();

  auto batch2 = stack_tensor(stack, 2);
  auto batch2_dtype = batch2.scalar_type();

  if (beta == 0.0) {
    return AlphaSharedMeta(batch1_dtype, batch2_dtype, alpha);
  }

  auto meta = AlphaSharedMeta(batch1_dtype, batch2_dtype, alpha);
  meta.push_back(BetaSharedMeta(input_rank, input_dtype));

  SharedMetaTensor common_data = {3, batch1_dtype};

  SharedMetaData add{"add_fwd"};
  add.inputs_data = {2, common_data};
  add.outputs_data = {common_data};
  meta.push_back(add);

  return meta;
}

void Baddbmm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto meta = BaddbmmMeta(stack)[0];
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1), syn_in(2)};
  auto baddbmm_out = BaddbMMCommon(this, graph, stack, input_tensor, meta);

  syn_out(0) = std::move(baddbmm_out[0]);
}
} // namespace habana
