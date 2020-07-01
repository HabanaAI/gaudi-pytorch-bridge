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

class BatchNormForwardOperator : public habana::HabanaOperator {
 public:
  // NOTE: BatchNormForwardOperator node_type differs for training and eval
  BatchNormForwardOperator(
      int device_id,
      c10::ScalarType scalarType,
      std::string node_type)
      : HabanaOperator(node_type) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
    // assign layouts for input and output tensors

    kernel_meta_data_.input_layout.assign({habana::LayoutFormat::NHWC,
                                           habana::LayoutFormat::ANY,
                                           habana::LayoutFormat::ANY,
                                           habana::LayoutFormat::ANY,
                                           habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC,
                                            habana::LayoutFormat::ANY,
                                            habana::LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);

  virtual std::vector<at::Tensor> preProcessInputs(torch::jit::Stack& inputs);

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  c10::ScalarType scalarType_;
  std::vector<synapse_helpers::tensor_or_ref> tensors_;
};

class BatchNormBackwardOperator : public habana::HabanaOperator {
 public:
  // NOTE: BatchNormBackwardOperator node_type differs for training and eval
  BatchNormBackwardOperator(
      int device_id,
      c10::ScalarType scalarType,
      std::string node_type)
      : HabanaOperator(node_type) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
    // assign layouts for input and output tensors
    kernel_meta_data_.input_layout.assign({habana::LayoutFormat::NHWC,
                                           habana::LayoutFormat::ANY,
                                           habana::LayoutFormat::ANY,
                                           habana::LayoutFormat::ANY,
                                           habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC,
                                            habana::LayoutFormat::ANY,
                                            habana::LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);

  virtual std::vector<at::Tensor> preProcessInputs(torch::jit::Stack& inputs);

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  c10::ScalarType scalarType_;
  std::vector<synapse_helpers::tensor_or_ref> tensors_;
};
