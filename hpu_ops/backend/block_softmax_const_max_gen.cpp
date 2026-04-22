/**
 * Copyright (c) 2025 Intel Corporation
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

#include "generated/backend/block_softmax_const_max.h"

namespace habana {

namespace {
at::ScalarType getOutDtype(const at::Stack& stack) {
  return stack[6].toOptional<c10::ScalarType>().value_or(
      at::ScalarType::BFloat16);
}

int roundUpToMultipleOf(int x, int multiplier) {
  return ((x + multiplier - 1) / multiplier) * multiplier;
}
} // namespace

FillParamsT BlockSoftmaxConstMaxParams(const at::Stack& stack) {
  const auto batch_size = stack.at(3).toInt();
  const auto global_block_max = stack.at(4).toDouble();
  const auto output_scale = stack.at(5).toDouble();
  const auto out_dtype = getOutDtype(stack);
  const auto mode = out_dtype == at::ScalarType::BFloat16
      ? BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_BF16_OUT
      : BLOCK_SOFTMAX_CONSTANT_MAXER_MODE_HF8_OUT;

  PARAMS_STUB(ns_BlockSoftmaxConstantMax::Params);
  params->batchSize = batch_size;
  params->outputScale = output_scale;
  params->globalBlockMax = global_block_max;
  params->mode = mode;
  return paramsT;
}

OutputMetaDataVector BlockSoftmaxConstMaxMeta(const at::Stack& stack) {
  const auto out_dtype = getOutDtype(stack);
  static constexpr int input_dim = 5;
  static constexpr int block_size = 128;

  TORCH_CHECK(
      out_dtype == at::ScalarType::BFloat16 ||
          out_dtype == at::ScalarType::Float8_e4m3fn,
      "Supported output dtypes are bfloat16 and float8_e4m3, got ",
      out_dtype);

  const auto& attn = stack_tensor(stack, 0);
  const auto attn_shape = attn.sizes();

  TORCH_CHECK(
      attn_shape.size() == input_dim,
      "attn tensor must be ",
      input_dim,
      "-dimensional, got ",
      attn_shape.size());

  TORCH_CHECK(
      attn_shape[3] == 1,
      "attn tensor must have size 1 at dimension 3, got ",
      attn_shape[3]);

  TORCH_CHECK(
      attn_shape[4] == block_size,
      "attn tensor must have size ",
      block_size,
      " at dimension 4, got ",
      attn_shape[4]);

  const auto block_bias_shape = stack_tensor(stack, 1).sizes();

  TORCH_CHECK(
      block_bias_shape.size() == input_dim,
      "block_bias tensor must be ",
      input_dim,
      "-dimensional, got ",
      block_bias_shape.size());

  TORCH_CHECK(
      block_bias_shape[0] == attn_shape[0],
      "block_bias tensor and attn tensor must have the same size at dimension 0, got ",
      block_bias_shape[0],
      " and ",
      attn_shape[0]);

  TORCH_CHECK(
      block_bias_shape[4] == attn_shape[4],
      "block_bias tensor and attn tensor must have the same size at dimension 4, got ",
      block_bias_shape[4],
      " and ",
      attn_shape[4]);

  TORCH_CHECK(
      block_bias_shape[1] == 1 && block_bias_shape[2] == 1 &&
          block_bias_shape[3] == 1,
      "block_bias tensor must have size 1 at dimensions 1, 2 and 3, got ",
      block_bias_shape[1],
      ", ",
      block_bias_shape[2],
      " and ",
      block_bias_shape[3]);

  const auto block_groups_shape = stack_tensor(stack, 2).sizes();

  TORCH_CHECK(
      block_groups_shape.size() == 1,
      "block_groups tensor must be 1-dimensional, got ",
      block_groups_shape.size());

  TORCH_CHECK(
      block_groups_shape[0] == attn_shape[0],
      "block_groups tensor and attn tensor must have the same size at dimension 0, got ",
      block_groups_shape[0],
      " and ",
      attn_shape[0]);

  OutputMetaDataVector metaVec(1);
  auto& meta = metaVec.front();
  meta.shape = attn_shape.vec();
  meta.dtype = out_dtype;

  return metaVec;
}

SharedMetaDataVector BlockSoftmaxConstMaxSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode /*unused*/) {
  const auto& attn = stack_tensor(stack, 0);
  const auto& block_bias = stack_tensor(stack, 1);
  const auto& block_groups = stack_tensor(stack, 2);
  const auto input_dtype = attn.scalar_type();

  SharedMetaDataVector vec;
  vec.reserve(2);

  auto& meta_stage1 = vec.emplace_back("block_softmax_constant_max_stage1");
  meta_stage1.inputs_data.emplace_back(attn.dim(), input_dtype);
  meta_stage1.inputs_data.emplace_back(
      block_bias.dim(), block_bias.scalar_type());
  meta_stage1.inputs_data.emplace_back(
      block_groups.dim(), block_groups.scalar_type());

  meta_stage1.outputs_data.emplace_back(3, input_dtype);
  meta_stage1.outputs_data.emplace_back(2, input_dtype);
  meta_stage1.outputs_data.emplace_back(3, input_dtype);

  auto& meta_stage2 = vec.emplace_back("block_softmax_constant_max_stage2");
  meta_stage2.inputs_data.emplace_back(3, input_dtype);
  meta_stage2.inputs_data.emplace_back(2, input_dtype);
  meta_stage2.inputs_data.emplace_back(
      block_groups.dim(), block_groups.scalar_type());
  meta_stage2.inputs_data.emplace_back(3, input_dtype);

  meta_stage2.outputs_data.emplace_back(attn.dim(), getOutDtype(stack));

  return vec;
}

void BlockSoftmaxConstMax::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "BlockSoftmaxConstMax::AddNode");
  const auto& attn = stackGetter.getNextInput<TensorsPair>();
  const auto& block_bias = stackGetter.getNextInput<TensorsPair>();
  const auto& block_groups = stackGetter.getNextInput<TensorsPair>();
  const auto batch_size = stackGetter.getNextInput<long>();

  const auto input_shape = attn.pt_t.sizes();
  const auto num_blocks = input_shape[0];
  const auto kv_heads = input_shape[1];
  const auto gqa = input_shape[2];
  const auto block_size = input_shape[4];

  static const int num_tpc =
      habana::HPUDeviceContext::get_device().type() == synDeviceGaudi3 ? 64
                                                                       : 24;
  static constexpr int vec_size = 128;

  std::vector<int64_t> attn_exp_out_shape{
      num_blocks, kv_heads * gqa, block_size};
  std::vector<int64_t> block_bias_reshape_out_shape{num_blocks, block_size};
  std::vector<int64_t> block_sum_per_tpc_out_shape{
      batch_size, num_tpc, roundUpToMultipleOf(gqa * kv_heads, vec_size)};

  const auto& params = BlockSoftmaxConstMaxParams(stack);

  auto stage1 = BuildOp(
      graph,
      "block_softmax_constant_max_stage1_bf16",
      {attn.syn_t, block_bias.syn_t, block_groups.syn_t},
      {{attn_exp_out_shape, at::ScalarType::BFloat16},
       {block_bias_reshape_out_shape, at::ScalarType::BFloat16},
       {block_sum_per_tpc_out_shape, at::ScalarType::BFloat16}},
      params.ptr(),
      params.size());

  const auto meta = OutputMeta(stack)[0];

  auto stage2 = BuildOp(
      graph,
      "block_softmax_constant_max_stage2_bf16",
      {stage1[0].get(), stage1[1].get(), block_groups.syn_t, stage1[2].get()},
      {{meta.shape, meta.dtype, 0}},
      params.ptr(),
      params.size());
  syn_out(0) = std::move(stage2[0]);
}

} // namespace habana
