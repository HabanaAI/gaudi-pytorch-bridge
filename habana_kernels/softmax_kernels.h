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
            "logsoftmax_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

class LogSoftmaxBackwardOperator : public HabanaOperator {
 public:
  LogSoftmaxBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "logsoftmax_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    // For logsoftmax_bwd_ kernel, the node inputs are in order {grad, output,
    // input} The synapse graph needs only the grad and output, and in the order
    // {output, grad}. p_context_->pt_inputs_ and p_context_->syn_inputs_ are
    // modified here to ensure this.
    kernel_meta_data_.tpc_input_order = {1, 0};
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  ;
};

// Sofmax Operator
//
class SoftmaxOperator : public HabanaOperator {
 public:
  SoftmaxOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "softmax_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

// Sofmax Operator
//
class SoftmaxIntOperator : public HabanaOperator {
 public:
  SoftmaxIntOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "softmax_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

// SoftmaxBackward Operator
//
class SoftmaxBackwardOperator : public HabanaOperator {
 public:
  SoftmaxBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "softmax_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    // For softmax_bwd_ kernel, the node inputs are in order {grad, output,
    // input} The synapse graph needs only the grad and output, and in the order
    // {output, grad} The p_context_->pt_inputs_ and p_context_->syn_inputs_
    // need to be modified to ensure this.
    kernel_meta_data_.tpc_input_order = {1, 0};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

};

} // end namespace habana
