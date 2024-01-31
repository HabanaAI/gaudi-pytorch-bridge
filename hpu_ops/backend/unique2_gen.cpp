/*******************************************************************************
 * Copyright (C) 2024 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/unique2.h"

namespace habana {

Unique2Eager::Unique2Eager(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0, 0, 0, 0}, {}, {}, false) {
  SetOutputMetaFn(UniqueMeta);
}

std::vector<synapse_helpers::tensor> UniqueCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    Unique2Params_t self_params,
    synTensor self_synin,
    c10::optional<int> final_result_index_0,
    c10::optional<int> final_result_index_1,
    [[maybe_unused]] c10::optional<int> final_result_index_2,
    [[maybe_unused]] c10::optional<int> final_result_index_3) {
  int elements = self_params.numel;
  std::vector<int64_t> output_shape{elements};
  std::vector<int64_t> valid_count_shape{1};
  ns_UniqueKernel::ParamsV2 params = {};
  params.sorted = self_params.sorted;
  params.returnCounts = self_params.return_counts;
  params.returnInverse = self_params.return_inverted;
  params.dim = -5; /// NOTE - arbitrary value based on the documentation
  std::vector<synTensor> inputs = {self_synin};
  auto guid = get_guid_with_precision("unique_fwd", self_params.dtype);
  auto shape_tensor_dtype =
      (common::IsInt64Supported() ? c10::ScalarType::Long
                                  : c10::ScalarType::Int);

  auto outputs = std::vector<NodeAttr::NodeOutputAttr>{
      {output_shape, self_params.dtype, final_result_index_0},
      {valid_count_shape, shape_tensor_dtype, final_result_index_1},
  };

  if (self_params.return_counts && self_params.return_inverted) {
    outputs.push_back(
        {output_shape, c10::ScalarType::Long, final_result_index_2});
    outputs.push_back(
        {output_shape, c10::ScalarType::Long, final_result_index_3});
  } else if (!self_params.return_counts != !self_params.return_inverted) {
    outputs.push_back(
        {output_shape, c10::ScalarType::Long, final_result_index_2});
  }

  return OpBackend::BuildNode(
      op, graph, {guid, inputs, outputs, &params, sizeof(params)});
}

void Unique2Eager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  auto self = stack_tensor(stack, 0);
  auto sorted = stack.at(1).toBool();
  auto inverted = stack.at(2).toBool();
  auto counts = stack.at(3).toBool();

  Unique2Params_t self_params;
  self_params.dtype = self.scalar_type();
  self_params.sizes = self.sizes().vec();
  self_params.numel = self.numel();
  self_params.sorted = sorted;
  self_params.return_inverted = inverted;
  self_params.return_counts = counts;

  auto unique = UniqueCommon(this, graph, self_params, syn_in(0), 0, 1, 2, 3);

  syn_out(0) = std::move(unique.at(0));
  syn_out(1) = std::move(unique.at(1));
  if (inverted != counts) {
    syn_out(2) = std::move(unique.at(2));
  } else if (inverted && counts) {
    syn_out(2) = std::move(unique.at(2));
    syn_out(3) = std::move(unique.at(3));
  }
}
} // namespace habana

static const auto& UniqueKernelRegistry = habana::KernelRegistry().add(
    "hpu::_unique2_eager",
    KERNEL_FN_GLOBAL(habana::Unique2Eager));
