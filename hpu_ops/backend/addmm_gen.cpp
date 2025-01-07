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

#include "generated/backend/_addmm_activation.h"
#include "generated/backend/addbmm.h"
#include "generated/backend/addmm.h"
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

sizes_vec AddMMOutshape(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto mat1 = stack_tensor(stack, 1);
  auto mat2 = stack_tensor(stack, 2);
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
  // Take output tensor dtype
  c10::optional<at::Tensor> output_tensor = c10::nullopt;
  c10::optional<c10::ScalarType> output_type = c10::nullopt;
  if (stack.at(stack.size() - 1).isTensor()) {
    output_tensor = stack.at(stack.size() - 1).toTensor();
    output_type = stack.at(stack.size() - 1).toTensor().scalar_type();
  }
  meta.dtype = habana_helpers::DTypeHelper::get_compute_dtype(
      stack,
      output_tensor,
      habana_helpers::DTypeHelper::DtypePromoteVariant::kPromoteToCommon,
      false,
      output_type);
  meta.shape = AddMMOutshape(stack)[0];
  return {meta};
}

OutputMetaDataVector AddBMMMeta(const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto batch1 = stack_tensor(stack, 1);
  auto batch2 = stack_tensor(stack, 2);
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
    return HandleReduction(
        op,
        graph,
        gemm_out[0].get(),
        "reduce_sum_multi_dim_fwd",
        {0} /*dimsToReduce*/,
        3 /*inputRank*/,
        false /*keepdim*/,
        {{output_shape, op->ScalarType(), final_idx}});
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

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

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

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

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

void AddMMActivation::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto meta = AddMMMeta(stack)[0];

  const float beta_val = stack.at(3).toScalar().toFloat();
  const float alpha_val = stack.at(4).toScalar().toFloat();

  const bool shouldUseParams = beta_val == 0.0 || beta_val == 1.0 ||
      alpha_val == 1.0 || alpha_val == 0.0;

  const bool append_activation = !(alpha_val == 0 && beta_val == 0);

  std::vector<synapse_helpers::tensor> result;
  if (shouldUseParams) {
    ns_AddmmKernel::Params params{};
    params.alpha = alpha_val;
    params.beta = beta_val;

    result = BuildOp(
        graph,
        guid_,
        {syn_in(0), syn_in(1), syn_in(2)},
        {{meta.shape,
          meta.dtype,
          append_activation ? c10::nullopt : c10::optional(0)}},
        &params,
        sizeof(params));
  } else {
    auto alpha_tensor = ConstantHelper(graph, alpha_val, ScalarType(), 1);
    auto beta_tensor = ConstantHelper(graph, beta_val, ScalarType(), 1);
    result = BuildOp(
        graph,
        guid_,
        {syn_in(0),
         syn_in(1),
         syn_in(2),
         beta_tensor.get(),
         alpha_tensor.get()},
        {{meta.shape,
          meta.dtype,
          append_activation ? c10::nullopt : c10::optional(0)}});
  }

  if (append_activation) {
    bool use_gelu = stack.at(5).toBool();
    std::vector<NodeAttr::NodeOutputAttr> act_output_attr{
        {meta.shape, meta.dtype, 0}};
    if (use_gelu) {
      act_output_attr.push_back({meta.shape, meta.dtype});
    }
    auto act = BuildOp(
        graph,
        get_guid_with_precision(use_gelu ? "gelu_fwd" : "relu_fwd", meta.dtype),
        {result[0].get()},
        act_output_attr);
    syn_out(0) = std::move(act[0]);
  } else {
    syn_out(0) = std::move(result[0]);
  }
}

namespace {

SharedMetaDataVector BetaSharedMeta(
    const int input_rank,
    const at::ScalarType& input_dtype) {
  SharedMetaData constantSharedMeta{"constant"};
  constantSharedMeta.outputs_data.emplace_back(2, input_dtype);

  SharedMetaData mul{"mult_fwd"};
  mul.inputs_data = {{input_rank, input_dtype}, {1, input_dtype}};
  mul.outputs_data = {mul.inputs_data[0]};
  return {mul, constantSharedMeta};
}

SharedMetaDataVector AlphaSharedMeta(
    const at::ScalarType& input_dtype,
    const at::ScalarType& batch1_dtype,
    const at::ScalarType& batch2_dtype,
    const float alpha) {
  SharedMetaTensor meta_3d_1{3, batch1_dtype};
  SharedMetaTensor meta_2d_1{2, batch1_dtype};

  SharedMetaData bmm{"batch_gemm"};
  bmm.inputs_data = {meta_3d_1, {3, batch2_dtype}};
  bmm.outputs_data = {meta_3d_1};

  SharedMetaData reduce{"reduce_sum_multi_dim_fwd"};
  reduce.inputs_data = {meta_3d_1};
  reduce.outputs_data = {meta_2d_1};

  SharedMetaDataVector meta{bmm, reduce};

  if (alpha != 1.0) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(2, input_dtype);
    meta.push_back(constantSharedMeta);

    SharedMetaData mul{"mult_fwd"};
    mul.inputs_data = {meta_2d_1, {1, batch1_dtype}};
    mul.outputs_data = {meta_2d_1};
    meta.push_back(mul);
  }

  return meta;
}

} // namespace

SharedMetaDataVector AddBMMSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const float beta = stack.at(3).toScalar().toFloat();
  const float alpha = stack.at(4).toScalar().toFloat();

  const auto& input = stack_tensor(stack, 0);
  const auto input_dtype = input.scalar_type();
  if (alpha == 0.0 and beta == 0.0) {
    SharedMetaData constantSharedMeta{"constant"};
    constantSharedMeta.outputs_data.emplace_back(2, input_dtype);
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

  SharedMetaData add{"add_fwd"};
  add.inputs_data = {{2, batch1_dtype}, {1, batch1_dtype}};
  add.outputs_data = {add.inputs_data[0]};
  meta.push_back(add);

  return meta;
}

void AddBMM::AddNode(synapse_helpers::graph& graph, const at::Stack& stack) {
  auto outshape = AddBMMMeta(stack)[0].shape;
  std::vector<synTensor> input_tensor{syn_in(0), syn_in(1), syn_in(2)};
  const int64_t batch_size = stack_tensor(stack, 1).sizes()[0];
  auto gemm_outshape = {batch_size, outshape[0], outshape[1]};
  auto addbmm_out = AddMMCommon(
      this, graph, stack, input_tensor, outshape, gemm_outshape, true);

  syn_out(0) = std::move(addbmm_out[0]);
}
} // namespace habana
