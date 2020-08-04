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
using namespace torch;

//
// Gather2d Operator
class Gather2dOperator : public HabanaOperator {
 public:
  Gather2dOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Slice Operator
class SliceOperator : public HabanaOperator {
 public:
  SliceOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("slice") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  void SetPTOutputs(const torch::jit::Stack& inputs);

  Tensor AllocateOutputTensor(
      const Tensor& self,
      int64_t& dim,
      int64_t& start,
      int64_t& end,
      int64_t& step,
      bool is_output_persistent);
};

//
// Narrow Operator
class NarrowOperator : public SliceOperator {
 public:
  NarrowOperator(int device_id, c10::ScalarType scalarType)
      : SliceOperator(device_id, scalarType) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// Gather Operator
//
class GatherOperator : public HabanaOperator {
 public:
  GatherOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "gather_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  Tensor AllocateOutput(torch::jit::Stack& inputs);
};

//
// IndexSelect Operator
class IndexSelectOperator : public GatherOperator {
 public:
  IndexSelectOperator(int device_id, c10::ScalarType scalarType)
      : GatherOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

 private:
  Tensor AllocateOutputTensor(const Tensor& self, int64_t& dim, int64_t& index);
};

//
// Select Operator
class SelectOperator : public HabanaOperator {
 public:
  SelectOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("select") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  void SetPTOutputs(const torch::jit::Stack& inputs);
};

// Scatter Operator
class ScatterOperator : public HabanaOperator {
 public:
  ScatterOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "scatter_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// IndexPutOperator
class IndexPutOperator : public HabanaOperator {
 public:
  IndexPutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "index_put_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) final;

 protected:
  c10::ScalarType scalarType_;
};

//
// Arange Operator
class ArangeOperator : public HabanaOperator {
 public:
  ArangeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "range_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  Tensor AllocateOutput(torch::jit::Stack& inputs);
};
