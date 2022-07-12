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
namespace habana {

uint32_t get_seed_hpu(const c10::optional<torch::Generator>& gen);
at::Tensor get_seed_tensor_hpu(const c10::optional<torch::Generator>& gen);

// Uniform Operator
class UniformOperator : public HabanaOperator {
 public:
  UniformOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "random_uniform_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Normal Operator
class NormalOperator : public HabanaOperator {
 public:
  NormalOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "random_normal_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.tpc_input_order = {habana::NO_INPUTS};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Bernoulli Operator
class BernoulliOperator : public HabanaOperator {
 public:
  BernoulliOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "random_bernoulli_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// BernoulliScalar Operator
class BernoulliScalarOperator : public HabanaOperator {
 public:
  BernoulliScalarOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "random_bernoulli_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Bernoulli Operator
class DropoutOperator : public HabanaOperator {
 public:
  DropoutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "dropout_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static at::Tensor GenerateAndCopySeedToHPU(
      torch::jit::Stack& inputs,
      bool is_persistent);
  void SetPTOutputs(
      const torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);

  virtual DMAInputGeneratorType getDMAInputGeneratorType() override {
    return DMAInputGeneratorType::SEEDTENSOR;
  }

  using HabanaOperator::SetPTOutputs;
};

// RandShuffle Operator
class RandomShuffleOperator : public HabanaOperator {
 public:
  RandomShuffleOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "random_shuffle_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// RandPerm Operator
class RandpermOperator : public HabanaOperator {
 public:
  RandpermOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "randperm_" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  static at::Tensor GenerateAndCopySeedToHPU(
      torch::jit::Stack& inputs,
      bool is_persistent);
  DMAInputGeneratorType getDMAInputGeneratorType() override {
    return DMAInputGeneratorType::SEEDTENSOR;
  }
};
} // namespace habana
