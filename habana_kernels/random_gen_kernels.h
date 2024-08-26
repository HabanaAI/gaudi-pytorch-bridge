/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
namespace habana {

at::Generator& getDefaultHPUGenerator();
at::Generator createHPUGenerator();
uint32_t get_seed_hpu(const c10::optional<torch::Generator>& gen);
at::Tensor get_seed_tensor_hpu(const c10::optional<torch::Generator>& gen);

// RandShuffle Operator
class RandomShuffleOperator : public HabanaOperator {
 public:
  RandomShuffleOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("random_shuffle_fwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// RandPerm Operator
class RandpermOperator : public HabanaOperator {
 public:
  RandpermOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("randperm", scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class RandpermOperatorHT : public RandpermOperator {
 public:
  RandpermOperatorHT(int device_id, c10::ScalarType scalarType)
      : RandpermOperator(device_id, scalarType) {}

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class HabanaRandomSeedOperator : public habana::OpBackend {
 public:
  HabanaRandomSeedOperator(int device_id, c10::ScalarType scalar_type)
      : OpBackend(
            device_id,
            "habana_random_seed",
            scalar_type,
            {},
            {},
            {},
            false) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
