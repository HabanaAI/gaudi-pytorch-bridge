/**
 * Copyright (c) 2021-2025 Intel Corporation
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

LazyPermute1DSparseDataWithWeights::LazyPermute1DSparseDataWithWeights(
    int device_id,
    c10::ScalarType scalar_type)
    : LazyPermuteSparseDataCommon(device_id, scalar_type, true, true) {}

void LazyPermute1DSparseDataWithWeights::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  LazyPermuteSparseDataCommon::AddLazyPermuteSparseDataNode(graph, stack, true);
}

LazyPermute1DSparseDataWithoutWeights::LazyPermute1DSparseDataWithoutWeights(
    int device_id,
    c10::ScalarType scalar_type)
    : LazyPermuteSparseDataCommon(device_id, scalar_type, true, false) {}

void LazyPermute1DSparseDataWithoutWeights::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  LazyPermuteSparseDataCommon::AddLazyPermuteSparseDataNode(graph, stack, true);
}

LazyPermute2DSparseDataWithWeights::LazyPermute2DSparseDataWithWeights(
    int device_id,
    c10::ScalarType scalar_type)
    : LazyPermuteSparseDataCommon(device_id, scalar_type, false, true) {}

void LazyPermute2DSparseDataWithWeights::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  LazyPermuteSparseDataCommon::AddLazyPermuteSparseDataNode(
      graph, stack, false);
}

LazyPermute2DSparseDataWithoutWeights::LazyPermute2DSparseDataWithoutWeights(
    int device_id,
    c10::ScalarType scalar_type)
    : LazyPermuteSparseDataCommon(device_id, scalar_type, false, false) {}

void LazyPermute2DSparseDataWithoutWeights::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  LazyPermuteSparseDataCommon::AddLazyPermuteSparseDataNode(
      graph, stack, false);
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

using namespace std::literals;

LazyBoundsCheckIndices::LazyBoundsCheckIndices(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "bounds_check_indices_fwd_",
          scalar_type,
          {},
          {0, 1, 2},
          {},
          false) {}

void LazyBoundsCheckIndices::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "LazyBoundsCheckIndices::AddNode");
  auto indices = stackGetter.getNextInput<TensorsPair>();
  auto offsets = stackGetter.getNextInput<TensorsPair>();
  auto warning = stackGetter.getNextInput<TensorsPair>();
  auto rowsPerTable = stackGetter.getNextInput<TensorsPair>();
  auto boundsCheckMode = stackGetter.getNextInput<long>();
  auto weights = stackGetter.getNextInput<std::optional<TensorsPair>>();

  std::vector<synTensor> inputs = {
      rowsPerTable.syn_t, indices.syn_t, offsets.syn_t, warning.syn_t};
  if (weights) {
    inputs.push_back(weights.value().syn_t);
  }

  std::string guid =
      get_guid_with_precision("bounds_check_indices_fwd"sv, ScalarType());

  std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
      {indices.pt_t.sizes(), indices.pt_t.scalar_type(), 0},
      {offsets.pt_t.sizes(), offsets.pt_t.scalar_type(), 1},
      {warning.pt_t.sizes(), warning.pt_t.scalar_type(), 2}};

  ns_BoundsCheckIndicesKernel::Params params{};
  params.mode = static_cast<BoundsCheckMode_t>(boundsCheckMode);

  auto results = OpBackend::BuildNode(
      this, graph, {guid, inputs, output_attrs, &params, sizeof(params)});

  for (size_t i = 0; i < output_attrs.size(); ++i) {
    syn_out(i) = std::move(results[i]);
  }
}

LazySplitPermuteCat::LazySplitPermuteCat(
    int device_id,
    c10::ScalarType scalar_type)
    : OpBackend(
          device_id,
          "split_permute_cat",
          scalar_type,
          {0},
          {},
          {},
          false) {}

void LazySplitPermuteCat::AddNode(
    synapse_helpers::graph& graph,
    const at::Stack& stack) {
  StackGetter stackGetter(this, stack, "LazySplitPermuteCat::AddNode");
  auto input = stackGetter.getNextInput<TensorsPair>();
  auto indices = stackGetter.getNextInput<TensorsPair>();

  auto batchSize = stackGetter.getNextInput<long>();
  auto numFeatures = stackGetter.getNextInput<long>();
  auto dims = stackGetter.getNextInput<long>();

  std::string guid = get_guid_with_precision(
      "split_permute_cat_fwd"sv, input.pt_t.scalar_type());

  ns_SplitPermuteCat::Params params;
  HABANA_ASSERT(
      batchSize <= std::numeric_limits<int>::max(), "Max batch size exceeded");
  HABANA_ASSERT(
      numFeatures <= std::numeric_limits<int>::max(),
      "Max numFeatures exceeded");
  HABANA_ASSERT(dims <= std::numeric_limits<int>::max(), "Max dims exceeded");
  params.batchSize = static_cast<int>(batchSize);
  params.numFeatures = static_cast<int>(numFeatures);
  params.dims = static_cast<int>(dims);

  auto output = OpBackend::BuildNode(
      this,
      graph,
      {guid,
       {input.syn_t, indices.syn_t},
       {{input.pt_t.sizes(), input.pt_t.scalar_type(), 0}},
       &params,
       sizeof(params)});

  syn_out(0) = std::move(output[0]);
}

} // namespace habana

static const auto& FBGEMMKernelsKernelRegistry =
    habana::KernelRegistry()
        .REGISTER_HPU_BACKEND(
            "hpu::habana_permute_1D_sparse_data",
            habana::LazyPermute1DSparseDataWithWeights)
        .REGISTER_HPU_BACKEND(
            "hpu::habana_permute_1D_sparse_data_without_weights",
            habana::LazyPermute1DSparseDataWithoutWeights)
        .REGISTER_HPU_BACKEND(
            "hpu::habana_permute_2D_sparse_data",
            habana::LazyPermute2DSparseDataWithWeights)
        .REGISTER_HPU_BACKEND(
            "hpu::habana_permute_2D_sparse_data_without_weights",
            habana::LazyPermute2DSparseDataWithoutWeights)
        .REGISTER_HPU_BACKEND(
            "hpu::habana_expand_into_jagged_permute",
            habana::LazyExpandIntoJaggedPermute)
        .REGISTER_HPU_BACKEND(
            "hpu::habana_bounds_check_indices",
            habana::LazyBoundsCheckIndices)
        .REGISTER_HPU_BACKEND(
            "hpu::habana_split_permute_cat",
            habana::LazySplitPermuteCat);
