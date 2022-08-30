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
  virtual OutputShapeInfRetType ComputeOutputShape(
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
// MeanDimOutOperator Operator
class MeanDimOutOperator : public ReduceOperator {
 public:
  MeanDimOutOperator(int device_id, const std::string& guid)
      : ReduceOperator(device_id, guid) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// MeanDim Operator
class MeanDimOperator : public ReduceOperator {
 public:
  MeanDimOperator(int device_id, c10::ScalarType scalar_type)
      : ReduceOperator(
            device_id,
            "reduce_mean_fwd_" +
                habana_helpers::name_suffix_from_type(scalar_type)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// Mean Operator
class MeanOperator : public ReduceOperator {
 public:
  MeanOperator(int device_id, c10::ScalarType scalar_type)
      : ReduceOperator(
            device_id,
            "reduce_mean_fwd_" +
                habana_helpers::name_suffix_from_type(scalar_type)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs) override;
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// ProdDim Operator
class ProdDimOperator : public ReduceOperator {
 public:
  ProdDimOperator(int device_id, c10::ScalarType scalarType)
      : ReduceOperator(
            device_id,
            "reduce_prod_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// Prod Operator
class ProdOperator : public ReduceOperator {
 public:
  ProdOperator(int device_id, c10::ScalarType scalarType)
      : ReduceOperator(
            device_id,
            "reduce_prod_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}

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
            "reduce_sum_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}

  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs) override;
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
            "reduce_sum_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }
  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs);
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
            "reduce_sum_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs) override;
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
            "reduce_sum_square_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {}
};

//
// AnyDimOut Operator
class AnyDimOutOperator : public HabanaOperator {
 public:
  AnyDimOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_any" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// AnyDim Operator
class AnyDimOperator : public HabanaOperator {
 public:
  AnyDimOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_any" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Any Operator
class AnyOperator : public HabanaOperator {
 public:
  AnyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_any" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// _grad_sum_to_size Operator
class GradSumToSizeOperator : public HabanaOperator {
 public:
  GradSumToSizeOperator(int device_id, c10::ScalarType scalar_type)
      : HabanaOperator(
            "grad_sum_to_size_" +
            habana_helpers::name_suffix_from_type(scalar_type)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// ArgMax Operator
class ArgMaxOperator : public ReduceOperator {
 public:
  ArgMaxOperator(int device_id, c10::ScalarType scalarType)
      : ReduceOperator(
            device_id,
            "argmax_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
};

class AllOutOperator : public HabanaOperator {
 public:
  AllOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_all" + habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
// Reduce Sum Backward Operator.
class ReduceSumBwdOperator : public HabanaOperator {
 public:
  ReduceSumBwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_sum_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Reduce Mean Backward Operator.
class ReduceMeanBwdOperator : public HabanaOperator {
 public:
  ReduceMeanBwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_mean_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class ReduceMultiOutputOperator : public ReduceOperator {
 public:
  ReduceMultiOutputOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string variant)
      : ReduceOperator(
            device_id,
            "reduce_" + variant + "_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
