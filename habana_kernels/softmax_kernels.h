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

namespace habana {

// LogSofmax Operator
//
class LogSoftmaxOperator : public HabanaOperator {
 public:
  LogSoftmaxOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator(
          "logsoftmax_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
        synapse_helpers::graph& graph,
        torch::jit::Stack& inputs,
        bool is_output_persistent = false);
};

class LogSoftmaxBackwardOperator : public HabanaOperator {
 public:
  LogSoftmaxBackwardOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator(
          "logsoftmax_bwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
 ;
};

// Sofmax Operator
//
class SoftmaxOperator : public HabanaOperator {
 public:
  SoftmaxOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};

}//end namespace habana
