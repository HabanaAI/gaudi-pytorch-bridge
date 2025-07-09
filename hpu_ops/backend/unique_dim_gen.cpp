/*******************************************************************************
 * Copyright (C) 2025 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/unique_dim.h"

namespace habana {

UniqueDimEager::UniqueDimEager(int device_id, c10::ScalarType scalar_type)
    : OpBackend(device_id, {}, scalar_type, {0, 0, 0, 0}, {}, {}, false) {
  SetOutputMetaFn(UniqueDimMeta);
}

std::vector<synapse_helpers::tensor> UniqueCommon(
    OpBackend* op,
    synapse_helpers::graph& graph,
    UniqueDimParams_t self_params,
    synTensor self_synin) {
  auto output_shape = self_params.sizes;
  auto param_shape = std::vector<int64_t>{output_shape.at(self_params.dim)};
  std::vector<int64_t> valid_count_shape{1};
  ns_UniqueKernel::ParamsV2 params = {};
  params.sorted = self_params.sorted;
  params.returnCounts = self_params.return_counts;
  params.returnInverse = self_params.return_inverted;
  params.dim = self_params.dim;
  std::vector<synTensor> inputs = {self_synin};
  using namespace std::literals;
  auto guid = get_guid_with_precision("unique_fwd"sv, self_params.dtype);
  auto shape_tensor_dtype =
      (common::IsInt64Supported() ? c10::ScalarType::Long
                                  : c10::ScalarType::Int);

  auto outputs = std::vector<NodeAttr::NodeOutputAttr>{
      {output_shape, self_params.dtype, 0},
      {valid_count_shape, shape_tensor_dtype, 1},
  };

  if (self_params.return_counts && self_params.return_inverted) {
    outputs.push_back({param_shape, c10::ScalarType::Long, 2});
    outputs.push_back({param_shape, c10::ScalarType::Long, 3});
  } else if (self_params.return_counts != self_params.return_inverted) {
    outputs.push_back({param_shape, c10::ScalarType::Long, 2});
  }
  return OpBackend::BuildNode(
      op, graph, {guid, inputs, outputs, &params, sizeof(params)});
}

void UniqueDimEager::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  const auto self = stack_tensor(stack, 0);

  UniqueDimParams_t self_params;
  self_params.dtype = self.scalar_type();
  self_params.sizes = self.sizes().vec();
  self_params.numel = self.numel();
  self_params.dim = stack.at(1).toInt();
  self_params.sorted = stack.at(2).toBool();
  self_params.return_inverted = stack.at(3).toBool();
  self_params.return_counts = stack.at(4).toBool();

  auto unique = UniqueCommon(this, graph, self_params, syn_in(0));

  syn_out(0) = std::move(unique.at(0));
  syn_out(1) = std::move(unique.at(1));
  if (self_params.return_inverted != self_params.return_counts) {
    syn_out(2) = std::move(unique.at(2));
  } else if (self_params.return_inverted && self_params.return_counts) {
    syn_out(2) = std::move(unique.at(2));
    syn_out(3) = std::move(unique.at(3));
  }
}
} // namespace habana

static const auto& UniqueKernelRegistry =
    habana::KernelRegistry().REGISTER_HPU_BACKEND(
        "hpu::unique_dim_eager",
        habana::UniqueDimEager);
