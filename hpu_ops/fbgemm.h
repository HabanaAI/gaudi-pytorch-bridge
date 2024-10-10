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

#pragma once

#include "hpu_ops/hpu_op_helper.h"
#include "hpu_ops/op_backend.h"

namespace habana {

struct LazyPermuteSparseDataCommon : OpBackend {
  LazyPermuteSparseDataCommon(
      int device_id,
      c10::ScalarType scalar_type,
      bool is1D,
      bool hasWeights);

  void AddLazyPermuteSparseDataNode(
      synapse_helpers::graph& graph,
      const at::Stack& stack,
      bool is1D) {
    bool hasWeights = stack.size() > 3;

    std::vector<synTensor> inputs = {syn_in(0), syn_in(1), syn_in(2)};
    if (hasWeights)
      inputs.push_back(syn_in(3));

    auto lengths = stack.at(1).toTensor();
    auto indices = stack.at(2).toTensor();

    std::string guid = "permute_" +
        (is1D ? std::string("1D") : std::string("2D")) + "_sparse_data_fwd_f32";

    auto lengths_shape = lengths.sizes().vec();
    if (is1D && lengths.dim() == 2) {
      lengths_shape = {lengths_shape[0] * lengths_shape[1]};
    }

    std::vector<NodeAttr::NodeOutputAttr> output_attrs = {
        {lengths_shape, lengths.scalar_type(), 0},
        {indices.sizes(), indices.scalar_type(), 1}};
    if (hasWeights) {
      auto weights = stack.at(3).toTensor();
      output_attrs.push_back({weights.sizes(), weights.scalar_type(), 2});
    }

    auto permuted = OpBackend::BuildNode(
        this, graph, {guid, inputs, output_attrs, nullptr, 0});

    syn_out(0) = std::move(permuted[0]);
    syn_out(1) = std::move(permuted[1]);
    if (hasWeights)
      syn_out(2) = std::move(permuted[2]);
  }
};

struct LazyPermute1DSparseData : LazyPermuteSparseDataCommon {
  LazyPermute1DSparseData(
      int device_id,
      c10::ScalarType scalar_type,
      bool hasWeights);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct LazyPermute2DSparseData : LazyPermuteSparseDataCommon {
  LazyPermute2DSparseData(
      int device_id,
      c10::ScalarType scalar_type,
      bool hasWeights);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct LazyExpandIntoJaggedPermute : OpBackend {
  LazyExpandIntoJaggedPermute(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct LazyBoundsCheckIndices : OpBackend {
  LazyBoundsCheckIndices(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

struct LazySplitPermuteCat : OpBackend {
  LazySplitPermuteCat(int device_id, c10::ScalarType scalar_type);
  void AddNode(synapse_helpers::graph&, const at::Stack&) override;
};

} // namespace habana
