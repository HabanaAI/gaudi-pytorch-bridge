/*******************************************************************************
 * Copyright (C) 2020-2024 Habana Labs, Ltd. an Intel Company
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

//
// Cat Operator
class CatOperator : public habana::HabanaOperator {
  auto CreateParamsAndAddToContext(int64_t axis);

 public:
  CatOperator(int device_id, c10::ScalarType) : HabanaOperator("concat") {
    this->CreateSynContext(device_id);
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;

  at::Tensor CheckAllocateOutput(
      torch::jit::Stack& inputs,
      const habana::OutputMetaData& output_metadata,
      bool is_dry_run);
  static std::vector<int64_t> compute_output_shape(
      const at::TensorList tensors,
      int64_t dim_);
  static void validate_cat_tensor_dim_sizes(
      const std::vector<std::vector<int64_t>>* tensors,
      int64_t dim);
};

//
// Permute Operator
class PermuteOperator : public habana::HabanaOperator {
 public:
  PermuteOperator(int device_id, c10::ScalarType scalarType);
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor& in, const std::vector<int64_t>& dims);
  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
};

//
//
class PermuteCLOperator : public PermuteOperator {
 public:
  PermuteCLOperator(int device_id, c10::ScalarType scalarType)
      : PermuteOperator(device_id, scalarType) {}
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

//
// Reshape Operator
class ReshapeOperator : public habana::HabanaOperator {
 public:
  ReshapeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("reshape") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.changes_dims = true;
    this->setNoComputeFlag();
  }
  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

//
// Transpose Operator
class TransposeOperator : public habana::HabanaOperator {
 public:
  TransposeOperator(int device_id, c10::ScalarType scalarType);
  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor& self, int dim0_, int dim1_);
};

//
// aten::t operator implementattion
class TOperator : public TransposeOperator {
 public:
  TOperator(int device_id, c10::ScalarType scalarType)
      : TransposeOperator(device_id, scalarType) {}

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override {
    inputs.insert(inputs.begin() + 1, c10::IValue(0));
    inputs.insert(inputs.begin() + 2, c10::IValue(-1));
    return TransposeOperator::InferOutputMeta(inputs);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override {
    TORCH_CHECK(
        inputs.size() == 1, "aten::t Operation expects 1 arguments as input")
    inputs.insert(inputs.begin() + 1, c10::IValue(0));
    inputs.insert(inputs.begin() + 2, c10::IValue(-1));
    TransposeOperator::AllocateAndAddSynapseNode(
        graph, inputs, output_metadata);
  }
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor& self) {
    return TransposeOperator::compute_output_shape(self, 0, -1);
  }
};

// Broadcast Operator
class BroadcastOperator : public habana::HabanaOperator {
 public:
  BroadcastOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("broadcast") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

//
// split_with_size Operator
class SplitWithSizeOperator : public habana::HabanaOperator {
 public:
  SplitWithSizeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("split_with_size") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;

  static std::vector<std::vector<int64_t>> compute_output_shape(
      const at::Tensor& self,
      c10::IntArrayRef split_sizes,
      int64_t dim);

  void SetPTOutputs(torch::jit::Stack& inputs) override;
};

// View Operator
class ViewOperator : public ReshapeOperator {
 public:
  ViewOperator(int device_id, c10::ScalarType scalarType)
      : ReshapeOperator(device_id, scalarType) {}
  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};
