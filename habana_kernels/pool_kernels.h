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

// Pool Operator
//
class MaxPool2dWithIndicesOperator : public HabanaOperator {
 public:
  MaxPool2dWithIndicesOperator(int device_id, c10::ScalarType scalarType)
    : HabanaOperator("maxpool_2d_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)){
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC, LayoutFormat::NHWC});
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);
};

class MaxPool2dOperator : public MaxPool2dWithIndicesOperator {
 public:
  MaxPool2dOperator(int device_id, c10::ScalarType scalarType)
      : MaxPool2dWithIndicesOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::NHWC,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY,
                                           LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::NHWC});
    p_context_->excluded_output_indices_ = {0};
  }
};