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

//
// Cat Operator
class CatOutOperator : public ::habana::HabanaOperator {
 public:
  CatOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("concat") {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutput(torch::jit::Stack& inputs);

 private:
  void validate_tensor_dim_sizes(c10::List<at::Tensor> tensors, int64_t dim);
  int64_t CheckAllocateOutput(torch::jit::Stack& inputs);
};

class CatOperator : public CatOutOperator {
 public:
  CatOperator(int device_id, c10::ScalarType scalarType)
      : CatOutOperator(device_id, scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutput(torch::jit::Stack& inputs);

 private:
  at::Tensor CheckAllocateOutput(torch::jit::Stack& inputs);
};

//
// Permute Operator
class PermuteOperator : public ::habana::HabanaOperator {
 public:
  PermuteOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor& in, const std::vector<int64_t>& dims);
};

//
// Reshape Operator
class ReshapeOperator : public ::habana::HabanaOperator {
 public:
  ReshapeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("reshape") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.changes_dims = true;
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Transpose Operator
class TransposeOperator : public ::habana::HabanaOperator {
 public:
  TransposeOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor& self, int dim0_, int dim1_);
};

//
// aten::t operator implementattion
class TOperator : public TransposeOperator {
 public:
  TOperator(int device_id, c10::ScalarType scalarType)
      : TransposeOperator(device_id, scalarType) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent) override {
    TORCH_CHECK(
        inputs.size() == 1, "aten::t Operation expects 1 arguments as input")
    inputs.insert(inputs.begin() + 1, c10::IValue(0));
    inputs.insert(inputs.begin() + 2, c10::IValue(1));
    TransposeOperator::AllocateAndAddSynapseNode(
        graph, inputs, is_output_persistent);
  }
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor& self) {
    return TransposeOperator::compute_output_shape(self, 0, 1);
  }
};

// Broadcast Operator
class BroadcastOperator : public ::habana::HabanaOperator {
 public:
  BroadcastOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("broadcast") {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// split_with_size Operator
class SplitWithSizeOperator : public ::habana::HabanaOperator {
 public:
  SplitWithSizeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("split_with_size") {
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  void SetPTOutputs(torch::jit::Stack& inputs);
};

// Flatten Operator
class FlattenOperator : public ReshapeOperator {
 public:
  FlattenOperator(int device_id, c10::ScalarType scalarType)
      : ReshapeOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

// View Operator
class ViewOperator : public ReshapeOperator {
 public:
  ViewOperator(int device_id, c10::ScalarType scalarType)
      : ReshapeOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};