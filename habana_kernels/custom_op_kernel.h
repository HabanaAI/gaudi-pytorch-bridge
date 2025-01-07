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
