/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */

#pragma once
#include "backend/habana_operator.h"
#include "include/habanalabs/hpu_custom_op.h"

namespace habana {

// A class that represents user's torch custom op
// All user's torch ops with a single user tpc kernel will pass here
// when lowered to synapse graph.
class CustomOperator : public HabanaOperator {
 public:
  CustomOperator(
      int device_id,
      habana::custom_op::HabanaCustomOpDescriptor desc)
      : HabanaOperator(desc.getGuid()), op_desc_(desc) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

 private:
  custom_op::HabanaCustomOpDescriptor op_desc_;
};

} // namespace habana
