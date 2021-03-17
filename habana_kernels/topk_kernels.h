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
#include "habana_kernels/habana_operator.h"
using namespace habana;

//
// TopkOut Operator
class TopkOutOperator : public HabanaOperator {
 public:
  TopkOutOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    values_persistent = false;
    indices_persistent = false;
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;

 private:
  bool values_persistent;
  bool indices_persistent;
};

//
// Topk Operator
class TopkOperator : public TopkOutOperator {
 public:
  TopkOperator(int device_id, const std::string& guid)
      : TopkOutOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// Sort Operator
class SortOperator : public TopkOutOperator {
 public:
  SortOperator(int device_id, const std::string& guid)
      : TopkOutOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};
