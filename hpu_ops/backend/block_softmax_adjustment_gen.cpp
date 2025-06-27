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

#include <perf_lib_layer_params.h>
#include <cmath>
#include <string_view>
#include <vector>
#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/stack_getter.h"

namespace sh = synapse_helpers;

#define CEIL_TO_VEC_SIZE(num, roundup) \
  (((num) + ((roundup)-1)) & ~((roundup)-1))
namespace habana {

OutputMetaDataVector BlockSoftmaxAdjustmentMeta(const at::Stack& stack) {
  auto block_maxes = stack_tensor(stack, 0);
  auto out_shape = stack[4].toIntList().vec();

  OutputMetaData meta;
  meta.shape = out_shape;
  meta.dtype = block_maxes.scalar_type();

  return {meta};
}

FillParamsT BlockSoftmaxAdjustmentParams(const at::Stack& stack) {
  const auto batchSize = stack.at(3).toScalar().toInt();

  PARAMS_STUB(ns_BlockSoftmaxAdjustment::Params);
  params->batchSize = batchSize;
  return paramsT;
}

// Meta function for block_softmax
OutputMetaDataVector BlockSoftmaxMeta(const at::Stack& stack) {
  // Calculate output shapes
  auto attn = stack_tensor(stack, 0);
  const c10::ScalarType dtype = attn.scalar_type();

  const int64_t num_blocks = attn.size(0);
  const int64_t kv_heads = attn.size(1);
  const int64_t gqa = attn.size(2);
  const int64_t num_tokens = attn.size(3);

  OutputMetaData attn_meta, block_maxes_meta, block_sums_meta;

  // Output 1: attn - same shape as input attn
  attn_meta.shape = attn.sizes().vec();
  attn_meta.dtype = dtype;

  // Output 2 & 3: b_maxes, b_sums - reshape and align to vector size
  const int64_t vec_size = (dtype == c10::ScalarType::Float) ? 64 : 128;
  int64_t flat_size = kv_heads * gqa * num_tokens;
  int64_t aligned_flat_size = CEIL_TO_VEC_SIZE(flat_size, vec_size);

  std::vector<int64_t> reduced_shape = {num_blocks, aligned_flat_size};

  block_maxes_meta.shape = reduced_shape;
  block_maxes_meta.dtype = dtype;

  block_sums_meta.shape = reduced_shape;
  block_sums_meta.dtype = dtype;

  return {attn_meta, block_maxes_meta, block_sums_meta};
}

SharedMetaDataVector BlockSoftmaxAdjustmentSharedMeta(
    const at::Stack& stack,
    habana_helpers::HabanaExecutionMode) {
  const auto& block_maxes = stack_tensor(stack, 0);
  const auto& block_sums = stack_tensor(stack, 1);
  const auto& block_groups = stack_tensor(stack, 2);
  auto precision_type = block_maxes.scalar_type();

  SharedMetaData adjustment_meta{"block_softmax_adjustment"};
  adjustment_meta.inputs_data = {
      {block_maxes.dim(), precision_type},
      {block_sums.dim(), precision_type},
      {block_groups.dim(), at::ScalarType::Int},
      {1, at::ScalarType::Int}};
  adjustment_meta.outputs_data.emplace_back(block_maxes.dim(), precision_type);

  return {adjustment_meta};
}

using namespace std::literals;
class BlockSoftmaxAdjustmentOperator : public OpBackend {
 public:
  BlockSoftmaxAdjustmentOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            NO_TPC + "block_softmax_adjustment",
            scalar_type,
            {0},
            {}, // inplace ids
            {},
            false) {}

  void AddNode(sh::graph& graph, const at::Stack& stack) override;
};

void BlockSoftmaxAdjustmentOperator::AddNode(
    sh::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(
      this, stack, "BlockSoftmaxAdjustmentOperator::AddNode");
  auto block_maxes = stackGetter.getNextInput<TensorsPair>();
  auto block_sums = stackGetter.getNextInput<TensorsPair>();
  auto block_groups = stackGetter.getNextInput<TensorsPair>();
  auto batch_size = stackGetter.getNextInput<int>();
  auto out_shape = stack[4].toIntList().vec();

  ns_BlockSoftmaxAdjustment::Params adjustment_params;
  adjustment_params.batchSize = batch_size;

  std::string adjustment_guid = get_guid_with_precision(
      "block_softmax_adjustment"sv, block_maxes.pt_t.scalar_type());

  std::vector<synTensor> input = {
      block_maxes.syn_t, block_sums.syn_t, block_groups.syn_t};
  auto adjustment = BuildOp(
      graph,
      adjustment_guid,
      std::move(input),
      {NodeAttr::NodeOutputAttr{out_shape, block_maxes.pt_t.scalar_type(), 0}},
      &adjustment_params,
      sizeof(adjustment_params));
  syn_out(0) = std::move(adjustment.at(0));
}

} // namespace habana

static auto& BlockSoftmaxAdjustmentKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::block_softmax_adjustment",
        habana::BlockSoftmaxAdjustmentOperator);
