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
struct RepeatOperator : public HabanaOperator {
  RepeatOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("tile_fwd", scalarType)) {
    CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      at::IntArrayRef repeats);

  static std::vector<int64_t> compute_reshape_output(
      const at::Tensor& self,
      at::IntArrayRef repeats);
};

struct RepeatOperatorHT : public RepeatOperator {
  RepeatOperatorHT(int device_id, c10::ScalarType scalarType)
      : RepeatOperator(device_id, scalarType) {}

  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  std::vector<int64_t> ComputeRepeatShapefromH2DTensor(
      const at::Tensor& host_tensor);
};

class RepeatInlvOperator : public HabanaOperator {
 public:
  RepeatInlvOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("repeat_fwd", scalarType)) {
    CreateSynContext(device_id);
  }
  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& input,
      int64_t dim,
      int64_t out_size);
};
class RepeatInlvOperatorHT : public RepeatInlvOperator {
 public:
  RepeatInlvOperatorHT(int device_id, c10::ScalarType scalarType)
      : RepeatInlvOperator(device_id, scalarType) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  std::vector<int64_t> ComputeRepeatShapefromH2DTensor(
      const at::Tensor& host_tensor);

  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
};
} // namespace habana
