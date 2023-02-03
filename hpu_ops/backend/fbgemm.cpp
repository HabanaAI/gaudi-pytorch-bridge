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
            KERNEL_FN_ARG(LazyPermute2DSparseData, false));
