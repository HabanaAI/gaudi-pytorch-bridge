/******************************************************************************
 * Copyright (C) 2023 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#include "generated/backend/baddbmm.h"

namespace habana {

sizes_vec BaddbmmOutputShape(const at::Stack& stack) {
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
  std::vector<int64_t> output_size{bs, res_rows, res_cols};

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

  return {output_size};
}

static std::vector<synapse_helpers::tensor> ComputeGEMM(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const at::IntArrayRef gemm_output_shape,
    c10::optional<int> final_idx = c10::nullopt) {
  NodeAttr::NodeOutputAttr gemm_node_output_attr = {
      gemm_output_shape, op->ScalarType()};
  gemm_node_output_attr.final_result_index = final_idx;
  synGEMMParams matmul_params{};
  std::vector<synapse_helpers::tensor> gemm_out = OpBackend::BuildNode(
      op,
      graph,
      {op->GetGuid(),
       std::move(input_tensor),
       {gemm_node_output_attr},
       &matmul_params,
       sizeof(matmul_params)});
  return gemm_out;
}

static std::vector<synapse_helpers::tensor> ComputeBetaSide(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const float beta_val,
    c10::optional<int> final_idx = c10::nullopt) {
  synapse_helpers::tensor beta_tensor = OpBackend::BuildConstant(
      op, graph, beta_val, op->ScalarType(), output_shape);

  std::vector<synTensor> node_inputs{input_tensor.at(0), beta_tensor.get()};
  std::vector<synapse_helpers::tensor> beta_side_out = OpBackend::BuildNode(
      op,
      graph,
      {get_guid_with_precision("mult", op->ScalarType()),
       std::move(node_inputs),
       {{output_shape, op->ScalarType(), final_idx}}});

  return beta_side_out;
}

static std::vector<synapse_helpers::tensor> ComputeAlphaSide(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const at::IntArrayRef gemm_output_shape,
    const float alpha_val,
    c10::optional<int> final_idx = c10::nullopt) {
  c10::optional<int> is_gemm_final_node = c10::nullopt;
  if (alpha_val == 1.0) {
    is_gemm_final_node = final_idx;
  }

  std::vector<synapse_helpers::tensor> gemm_out = ComputeGEMM(
      op,
      graph,
      input_tensor,
      output_shape,
      gemm_output_shape,
      is_gemm_final_node);

  if (alpha_val == 1.0) {
    return gemm_out;
  } else {
    auto alpha_tensor = OpBackend::BuildConstant(
        op, graph, alpha_val, op->ScalarType(), output_shape);
    std::vector<synTensor> mul_node_inputs{
        gemm_out[0].get(), alpha_tensor.get()};
    std::vector<synapse_helpers::tensor> alpha_mul_out = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("mult", op->ScalarType()),
         std::move(mul_node_inputs),
         {{output_shape, op->ScalarType(), final_idx}}});
    return alpha_mul_out;
  }
}

static std::vector<synapse_helpers::tensor> BaddbMMCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const at::IntArrayRef gemm_output_shape) {
  std::vector<synapse_helpers::tensor> baddbmm_out;

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  if (alpha_val == 0.0 && beta_val == 0.0) {
    baddbmm_out.emplace_back(OpBackend::BuildConstant(
        op, graph, 0.0, op->ScalarType(), output_shape, 0));
  } else if (alpha_val == 0.0 && beta_val != 0.0) {
    baddbmm_out = ComputeBetaSide(
        op, graph, {input_tensor.at(0)}, output_shape, beta_val, 0);
  } else if (alpha_val != 0.0 && beta_val == 0.0) {
    baddbmm_out = ComputeAlphaSide(
        op,
        graph,
        {input_tensor.at(1), input_tensor.at(2)},
        output_shape,
        gemm_output_shape,
        alpha_val,
        0);
  } else {
    auto beta_out = ComputeBetaSide(
        op, graph, {input_tensor.at(0)}, output_shape, beta_val);
    auto alpha_out = ComputeAlphaSide(
        op,
        graph,
        {input_tensor.at(1), input_tensor.at(2)},
        output_shape,
        gemm_output_shape,
        alpha_val);
    std::vector<synTensor> add_node_inputs{
        beta_out[0].get(), alpha_out[0].get()};
    baddbmm_out = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("add", op->ScalarType()),
         std::move(add_node_inputs),
         {{output_shape, op->ScalarType(), 0}}});
  }
  return baddbmm_out;
}

void Baddbmm::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = BaddbmmOutputShape(stack)[0];
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1), syn_in(2)};
  const int64_t batch_size = stack_tensor(stack, 1).sizes()[0];
  auto gemm_outshape = {batch_size, outshape[1], outshape[2]};
  auto baddbmm_out =
      BaddbMMCommon(this, graph, stack, input_tensor, outshape, gemm_outshape);

  syn_out(0) = std::move(baddbmm_out[0]);
}
} // namespace habana
