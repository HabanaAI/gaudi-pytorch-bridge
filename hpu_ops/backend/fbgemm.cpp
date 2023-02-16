/******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include "hpu_ops/fbgemm.h"

namespace habana {

LazyPermuteSparseDataCommon::LazyPermuteSparseDataCommon(
    int device_id,
    c10::ScalarType scalar_type,
    bool is1D,
    bool hasWeights)
    : OpBackend(
          device_id,
          "permute_" + (is1D ? std::string("1D") : std::string("2D")) +
              "_sparse_data_fwd_",
          scalar_type,
          hasWeights ? std::vector<int>{1, 2, 3} : std::vector<int>{1, 2},
          {},
          {},
          false) {}

LazyPermute1DSparseData::LazyPermute1DSparseData(
    int device_id,
    c10::ScalarType scalar_type,
    bool hasWeights)
    : LazyPermuteSparseDataCommon(device_id, scalar_type, true, hasWeights) {}

LazyPermute2DSparseData::LazyPermute2DSparseData(
    int device_id,
    c10::ScalarType scalar_type,
    bool hasWeights)
    : LazyPermuteSparseDataCommon(device_id, scalar_type, false, hasWeights) {}

void LazyPermute1DSparseData::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  LazyPermuteSparseDataCommon::AddNode(graph, stack, true);
}

void LazyPermute2DSparseData::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  LazyPermuteSparseDataCommon::AddNode(graph, stack, false);
}

LazyExpandIntoJaggedPermute::LazyExpandIntoJaggedPermute(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "expand_into_jagged_permute",
          scalar_type,
          {1},
          {},
          {},
          false) {}

void LazyExpandIntoJaggedPermute::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};

  auto input_offsets = stack.at(1).toTensor();

  int64_t output_size = stack.at(3).toScalar().toInt();

  std::string guid = "expand_into_jagged_permute_fwd_i32";

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {{output_size}, input_offsets.scalar_type(), 0}};

  auto permuted = OpBackend::BuildNode(
      this, graph, {guid, inputs, output_attrs, nullptr, 0});

  syn_out(0) = std::move(permuted[0]);
}

} // namespace habana

static const auto& FBGEMMKernelsKernelRegistry =
    habana::KernelRegistry()
        .add(
            "hpu::habana_permute_1D_sparse_data",
            KERNEL_FN_ARG(LazyPermute1DSparseData, true))
        .add(
            "hpu::habana_permute_1D_sparse_data_without_weights",
            KERNEL_FN_ARG(LazyPermute1DSparseData, false))
        .add(
            "hpu::habana_permute_2D_sparse_data",
            KERNEL_FN_ARG(LazyPermute2DSparseData, true))
        .add(
            "hpu::habana_permute_2D_sparse_data_without_weights",
            KERNEL_FN_ARG(LazyPermute2DSparseData, false))
        .add(
            "hpu::habana_expand_into_jagged_permute",
            KERNEL_FN_GLOBAL(habana::LazyExpandIntoJaggedPermute));
