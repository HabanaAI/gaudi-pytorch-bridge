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
namespace habana {
// KlDiv Operator
class KlDivOperator : public HabanaOperator {
 public:
  KlDivOperator(const int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("kl_div", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t reduction);
};

} // namespace habana
