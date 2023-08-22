/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#pragma once
#include "backend/habana_operator.h"
namespace habana {

class FilterAndSqueezeOperator : public HabanaOperator {
 public:
  FilterAndSqueezeOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);

    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::BCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::BCN,
         synapse_helpers::layouts::SynapseLayoutFormat::BCN,
         synapse_helpers::layouts::SynapseLayoutFormat::CN});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class NMSOperator : public HabanaOperator {
 public:
  NMSOperator(int device_id, const std::string& guid) : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class PostNmsOperator : public HabanaOperator {
 public:
  PostNmsOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class HabanaNMSOperator : public HabanaOperator {
 public:
  HabanaNMSOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;
};

class BatchedNMSOperator : public HabanaOperator {
 public:
  BatchedNMSOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
