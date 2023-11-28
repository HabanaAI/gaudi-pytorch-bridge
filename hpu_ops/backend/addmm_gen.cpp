/*******************************************************************************
 * Copyright (C) 2021-2023 Habana Labs, Ltd. an Intel Company
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

#include "generated/backend/addbmm.h"
#include "generated/backend/addmm.h"
#define idxSelf 0
#define idxMat1 1
#define idxMat2 2
#define idxBatch1 1
#define idxBatch2 2
#define idxBeta 3
#define idxAlpha 4

namespace habana {

sizes_vec AddMMOutshape(const at::Stack& stack) {
  auto self = stack_tensor(stack, idxSelf);
  auto mat1 = stack_tensor(stack, idxMat1);
  auto mat2 = stack_tensor(stack, idxMat2);
  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 1 || self.dim() == 0,
      "addmm: Expected self to be 0-D, 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(
      mat1.dim() == 2,
      "addmm: Expected mat1 to be 2-D, but got ",
      mat1.dim(),
      "-D");
  TORCH_CHECK(
      mat2.dim() == 2,
      "addmm: Expected mat2 to be 2-D, but got ",
      mat2.dim(),
      "-D");
  TORCH_CHECK(
      mat1.sizes()[1] == mat2.sizes()[0],
      "Matrices sizes are not compatible to multiply them");
  // (n, m)@(m, p) -> (n, p)
  std::vector<int64_t> matMulShape = {mat1.sizes()[0], mat2.sizes()[1]};
  std::vector<int64_t> outshape = at::infer_size(self.sizes(), matMulShape);
  return {outshape};
}

OutputMetaDataVector AddMMMeta(const at::Stack& stack) {
  OutputMetaData meta;
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      c10::nullopt,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false);
  meta.shape = AddMMOutshape(stack)[0];
  return {meta};
}

OutputMetaDataVector AddBMMMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, idxSelf);
  auto batch1 = stack_tensor(stack, idxBatch1);
  auto batch2 = stack_tensor(stack, idxBatch2);
  TORCH_CHECK(
      self.dim() == 2 || self.dim() == 1 || self.dim() == 0,
      "addbmm: Expected self to be 0-D, 1-D or 2-D, but got ",
      self.dim(),
      "-D");
  TORCH_CHECK(
      batch1.dim() == 3,
      "addbmm: Expected batch1 to be 3-D, but got ",
      batch1.dim(),
      "-D");
  TORCH_CHECK(
      batch2.dim() == 3,
      "addbmm: Expected batch2 to be 3-D, but got ",
      batch2.dim(),
      "-D");
  std::vector<int64_t> outshape{
      batch1.sizes()[1], batch2.sizes()[2]}; // (b, n, m)@(b, m, p) -> (n, p)

  OutputMetaData meta;
  meta.shape = outshape;
  meta.dtype = self.scalar_type();
  return {meta};
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

static std::vector<synapse_helpers::tensor> ComputeGEMM(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const at::IntArrayRef gemm_output_shape,
    const bool is_batch,
    c10::optional<int> final_idx = c10::nullopt) {
  NodeAttr::NodeOutputAttr gemm_node_output_attr = {
      gemm_output_shape, op->ScalarType()};
  if (!is_batch)
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

  if (!is_batch) {
    return gemm_out;
  } else {
    ns_Reduction::Params reduce_params{};
    reduce_params.reductionDimension = 2;
    std::vector<synTensor> reduce_node_inputs{gemm_out[0].get()};
    auto reduce_out = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("reduce_sum_fwd", op->ScalarType()),
         std::move(reduce_node_inputs),
         {{{1, gemm_output_shape.at(1), gemm_output_shape.at(2)},
           op->ScalarType()}},
         &reduce_params,
         sizeof(reduce_params)});
    std::vector<synapse_helpers::tensor> reshape_out;
    reshape_out.emplace_back(OpBackend::BuildReshape(
        op,
        graph,
        reduce_out[0].get(),
        output_shape,
        op->ScalarType(),
        final_idx));
    return reshape_out;
  }
}

static std::vector<synapse_helpers::tensor> ComputeAlphaSide(
    OpBackend* op,
    synapse_helpers::graph& graph,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const at::IntArrayRef gemm_output_shape,
    const float alpha_val,
    const bool is_batch,
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
      is_batch,
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

static std::vector<synapse_helpers::tensor> AddMMCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    const at::Stack& stack,
    std::vector<synTensor> input_tensor,
    const at::IntArrayRef output_shape,
    const at::IntArrayRef gemm_output_shape,
    const bool is_batch) {
  std::vector<synapse_helpers::tensor> addmm_out;

  const float beta_val = stack.at(idxBeta).toScalar().toFloat();
  const float alpha_val = stack.at(idxAlpha).toScalar().toFloat();

  if (alpha_val == 0.0 && beta_val == 0.0) {
    addmm_out.emplace_back(OpBackend::BuildConstant(
        op, graph, 0.0, op->ScalarType(), output_shape, 0));
  } else if (alpha_val == 0.0 && beta_val != 0.0) {
    addmm_out = ComputeBetaSide(
        op, graph, {input_tensor.at(0)}, output_shape, beta_val, 0);
  } else if (alpha_val != 0.0 && beta_val == 0.0) {
    addmm_out = ComputeAlphaSide(
        op,
        graph,
        {input_tensor.at(1), input_tensor.at(2)},
        output_shape,
        gemm_output_shape,
        alpha_val,
        is_batch,
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
        alpha_val,
        is_batch);
    std::vector<synTensor> add_node_inputs{
        beta_out[0].get(), alpha_out[0].get()};
    addmm_out = OpBackend::BuildNode(
        op,
        graph,
        {get_guid_with_precision("add", op->ScalarType()),
         std::move(add_node_inputs),
         {{output_shape, op->ScalarType(), 0}}});
  }
  return addmm_out;
}

void AddMM::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  const auto meta = AddMMMeta(stack);

  const float beta_val = stack.at(idxBeta).toScalar().toFloat();
  const float alpha_val = stack.at(idxAlpha).toScalar().toFloat();

  const bool shouldUseParams = beta_val == 0.0 || beta_val == 1.0 ||
      alpha_val == 1.0 || alpha_val == 0.0;

  // Kernel precision type is based on the input1 of the addmm op,
  // because we want to support configuration: inputs(fp8), output(bf16/fp32).
  // Formula: out = beta * input0 + alpha * (input1 @ input2)
  // GEMM returns higher precision dtype, so input0 has to be (bf16/fp32).
  auto guid =
      get_guid_with_precision("addmm", stack_tensor(stack, 1).scalar_type());

  if (shouldUseParams) {
    ns_AddmmKernel::Params params{};
    params.alpha = alpha_val;
    params.beta = beta_val;

    auto addmv = BuildOp(
        graph,
        guid,
        {syn_in(0), syn_in(1), syn_in(2)},
        {{meta[0].shape, meta[0].dtype, 0}},
        &params,
        sizeof(params));
    syn_out(0) = std::move(addmv[0]);
  } else {
    auto alpha_tensor = ConstantHelper(graph, alpha_val, ScalarType(), 1);
    auto beta_tensor = ConstantHelper(graph, beta_val, ScalarType(), 1);
    auto addmm = BuildOp(
        graph,
        guid,
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         beta_tensor.get(),
         alpha_tensor.get()},
        {{meta[0].shape, meta[0].dtype, 0}});

    syn_out(0) = std::move(addmm[0]);
  }
}

void AddBMM::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = AddBMMMeta(stack)[0].shape;
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1), syn_in(2)};
  const int64_t batch_size = stack_tensor(stack, idxBatch1).sizes()[0];
  auto gemm_outshape = {batch_size, outshape[0], outshape[1]};
  auto addbmm_out = AddMMCommon(
      this, graph, stack, input_tensor, outshape, gemm_outshape, true);

  syn_out(0) = std::move(addbmm_out[0]);
}
} // namespace habana
