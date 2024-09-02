/******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include "hpu_ops/backend/reduction_template.h"

namespace habana {

/**
 * Base class for reduce operators such as Mean, Sum etc.
 * Should never be instantiated directly.
 */
class ReduceOperator : public HabanaOperator {
 public:
  ReduceOperator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }
  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;

  /**
   * @brief This function wraps any negative dims in the input dims List
   * to a positive value within valid range. Also sorts inputs dims in
   * ascending order.
   */
  void sort_dims(
      std::vector<int64_t>& in_dim,
      int64_t dim,
      int64_t dims_to_reduce);
  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      const at::IntArrayRef dim,
      const bool keepdim);

 private:
  /**
   * @brief This function creates a graph with 1 or more reduction nodes
   * (according to dims provided in "in_dim"). This is done because TPC
   * reduction kernels can reduce along only 1 dim at a time.Additionally
   * a reshape node maybe added as the last node to remove reduced dims
   * in keepdim = False case.
   */
  std::tuple<synapse_helpers::tensor_or_ref, synapse_helpers::tensor_or_ref>
  CreateReductionGraph(
      synapse_helpers::graph& graph,
      at::Tensor& pyt_tensor,
      const at::Tensor& output,
      synapse_helpers::tensor_or_ref syn_tensor_in,
      synapse_helpers::tensor_or_ref syn_tensor_out,
      c10::IntArrayRef in_dim,
      bool keepdim);
  int get_num_tpc_outputs();
};

//
// Mean Operator
class MeanOperator : public ReduceOperator {
 public:
  MeanOperator(int device_id, c10::ScalarType scalar_type)
      : ReduceOperator(
            device_id,
            get_guid_with_precision("reduce_mean_fwd", scalar_type)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// SumDimOutOperator Operator
class SumDimOutOperator : public ReduceOperator {
 public:
  SumDimOutOperator(int device_id, c10::ScalarType scalarType)
      : ReduceOperator(
            device_id,
            get_guid_with_precision("reduce_sum_fwd", scalarType)) {}

  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// SumDim Operator
class SumDimOperator : public ReduceOperator {
 public:
  SumDimOperator(int device_id, c10::ScalarType scalarType)
      : ReduceOperator(
            device_id,
            get_guid_with_precision("reduce_sum_fwd", scalarType)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// Sum Operator
class SumOperator : public ReduceOperator {
 protected:
  SumOperator(int device_id, const std::string& guid)
      : ReduceOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

 public:
  SumOperator(int device_id, c10::ScalarType scalarType)
      : SumOperator(
            device_id,
            get_guid_with_precision("reduce_sum_fwd", scalarType)) {}
  InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// Sum Square Operator
class SumSquareOperator : public SumOperator {
 public:
  SumSquareOperator(int device_id, c10::ScalarType scalarType)
      : SumOperator(
            device_id,
            get_guid_with_precision("reduce_sum_square_fwd", scalarType)) {}
};

// Reduce Sum Backward Operator.
class ReduceSumBwdOperator : public HabanaOperator {
 public:
  ReduceSumBwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("reduce_sum_bwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
};

// Reduce Mean Backward Operator.
class ReduceMeanBwdOperator : public HabanaOperator {
 public:
  ReduceMeanBwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("reduce_mean_bwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
};

class ReduceMultiOutputOperator : public ReduceOperator {
 public:
  ReduceMultiOutputOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string variant)
      : ReduceOperator(
            device_id,
            get_guid_with_precision("reduce_" + variant + "_fwd", scalarType)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
