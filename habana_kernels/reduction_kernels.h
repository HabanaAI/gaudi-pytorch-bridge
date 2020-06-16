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
// Reduce Operator
class ReduceOperator : public HabanaOperator {
 public:
  ReduceOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});

  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// MeanDimOutOperator Operator
class MeanDimOutOperator : public ReduceOperator {
 public:
  MeanDimOutOperator(int device_id, const std::string& guid)
   : ReduceOperator(device_id, guid) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// MeanDim Operator
class MeanDimOperator : public ReduceOperator {
 public:
  MeanDimOperator(int device_id, const std::string& guid)
   : ReduceOperator(device_id, guid) {
       kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
   }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Mean Operator
class MeanOperator : public ReduceOperator {
 public:
  MeanOperator(int device_id, const std::string& guid)
   : ReduceOperator(device_id, guid) {
       kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
   }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
