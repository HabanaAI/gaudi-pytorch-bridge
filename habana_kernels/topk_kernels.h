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
namespace habana {

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
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

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
  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};
} // namespace habana
