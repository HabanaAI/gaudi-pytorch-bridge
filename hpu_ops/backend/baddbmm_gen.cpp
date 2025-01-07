/**
* Copyright (c) 2021-2024 Intel Corporation
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

static SharedMetaDataVector BetaSharedMeta(
    const int input_rank,
    const at::ScalarType& input_dtype) {
  SharedMetaData constantSharedMeta{"constant"};
  constantSharedMeta.outputs_data.emplace_back(3, input_dtype);

  SharedMetaData mul{"mult_fwd"};
  mul.inputs_data = {{input_rank, input_dtype}, {3, input_dtype}};
  mul.outputs_data = {{3, input_dtype}};
  return {mul, constantSharedMeta};
}

static SharedMetaDataVector AlphaSharedMeta(
    const at::ScalarType& input_dtype,
    const at::ScalarType& batch1_dtype,
    const at::ScalarType& batch2_dtype,
    const float alpha) {
  SharedMetaTensor common_data = {3, batch1_dtype};
  SharedMetaData bmm{"batch_gemm"};
  bmm.inputs_data = {common_data, {3, batch2_dtype}};
  bmm.outputs_data = {common_data};

  if (alpha != 1.0) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(3, input_dtype);

    SharedMetaData mul{"mult_fwd"};
    mul.inputs_data = {2, common_data};
    mul.outputs_data = {common_data};
    return {bmm, constantSharedMeta, mul};
  }

  return {bmm};
}

SharedMetaDataVector BAddBMMSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const float beta = stack.at(3).toScalar().toFloat();
  const float alpha = stack.at(4).toScalar().toFloat();

  const auto& input = stack_tensor(stack, 0);
  const auto input_dtype = input.scalar_type();
  if (alpha == 0.0 and beta == 0.0) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(3, input_dtype);
    return {constantSharedMeta};
  }

  const auto input_rank = input.dim();

  if (alpha == 0.0) {
    return BetaSharedMeta(input_rank, input_dtype);
  }

  const auto& batch1 = stack_tensor(stack, 1);
  const auto batch1_dtype = batch1.scalar_type();

  const auto& batch2 = stack_tensor(stack, 2);
  const auto batch2_dtype = batch2.scalar_type();

  if (beta == 0.0) {
    return AlphaSharedMeta(input_dtype, batch1_dtype, batch2_dtype, alpha);
  }

  auto meta = AlphaSharedMeta(input_dtype, batch1_dtype, batch2_dtype, alpha);
  const auto betaVec = BetaSharedMeta(input_rank, input_dtype);
  meta.insert(std::end(meta), std::begin(betaVec), std::end(betaVec));
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
