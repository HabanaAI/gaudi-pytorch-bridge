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

class RoiAlignFwdOperator : public HabanaOperator {
 public:
  RoiAlignFwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("roialign_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});

    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::XR,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class RoiAlignBwdOperator : public HabanaOperator {
 public:
  RoiAlignBwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("roialign_backward", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class RoiAlignBwdImplOperator : public HabanaOperator {
 public:
  RoiAlignBwdImplOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("roialign_bwd", scalarType)) {
    this->CreateSynContext(device_id);

    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::VN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::NSB,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class QuadTreeFwdImplOperator : public HabanaOperator {
 public:
  QuadTreeFwdImplOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("quad_tree_fwd", scalarType)) {
    this->CreateSynContext(device_id);

    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::AB,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::BSN});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::BSN});
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
