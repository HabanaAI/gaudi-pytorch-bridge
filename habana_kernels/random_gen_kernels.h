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
#pragma once
#include "backend/habana_operator.h"
#include "hpu_ops/op_backend.h"
namespace habana {

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
