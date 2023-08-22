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
