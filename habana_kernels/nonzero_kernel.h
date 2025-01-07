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

// NonZero Operator
class NonZeroOperator : public HabanaOperator {
 public:
  NonZeroOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("non_zero_fwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(const at::Tensor& input);
  static float round_dims(const at::Tensor& input_tensor, int group_size);

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_st_shape(
      const at::Tensor& input_tensor);
};
} // namespace habana
