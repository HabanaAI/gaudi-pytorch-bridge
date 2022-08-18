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
#include <torch/script.h>
#include "../hpu_ops/hpu_op_helper.h"
#include "habana_kernels/habana_operator.h"

namespace habana {

// OptimizerSparseSgd Operator
class OptimizerSparseSgdOperator : public HabanaOperator {
 public:
  OptimizerSparseSgdOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_sparse_sgd_with_valid_count_2d_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerSparseAdagradOperator : public HabanaOperator {
 public:
  OptimizerSparseAdagradOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_sparse_adagrad_with_valid_count_2d_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerAdamwOperator : public HabanaOperator {
 public:
  OptimizerAdamwOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_adamw_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerAdagradOperator : public HabanaOperator {
 public:
  OptimizerAdagradOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_adagrad_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerFusedAdagradOperator : public HabanaOperator {
 public:
  OptimizerFusedAdagradOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_adagrad_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerSGDOperator : public HabanaOperator {
 public:
  OptimizerSGDOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_sgd_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerFusedEMAOperator : public HabanaOperator {
 public:
  OptimizerFusedEMAOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "dummy_fusedema_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerFusedSGDOperator : public HabanaOperator {
 public:
  OptimizerFusedSGDOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_sgd_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerSGDMomentumOperator : public HabanaOperator {
 public:
  OptimizerSGDMomentumOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_sgd_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class OptimizerFusedSGDMomentumOperator : public HabanaOperator {
 public:
  OptimizerFusedSGDMomentumOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "optimizer_sgd_bwd_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
