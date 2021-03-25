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

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);

 private:
  /**
   * @brief This function wraps any negative dims in the input dims List
   * to a positive value within valid range. Also sorts inputs dims in
   * ascending order.
   */
  void sort_dims(
      c10::List<int64_t>& in_dim,
      int64_t dim,
      int64_t dims_to_reduce);

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
      synapse_helpers::tensor_or_ref syn_tensor_in,
      synapse_helpers::tensor_or_ref syn_tensor_out,
      c10::IntArrayRef in_dim,
      bool keepdim,
      c10::ScalarType dtype);
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
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

//
// MeanDim Operator
class MeanDimOperator : public ReduceOperator {
 public:
  MeanDimOperator(int device_id, const std::string& guid)
      : ReduceOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

//
// Mean Operator
class MeanOperator : public ReduceOperator {
 public:
  MeanOperator(int device_id, const std::string& guid)
      : ReduceOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
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
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
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
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

//
// SumDimOutOperator Operator
class SumDimOutOperator : public ReduceOperator {
 public:
  SumDimOutOperator(int device_id, const std::string& guid)
      : ReduceOperator(device_id, guid) {}

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

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

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

//
// Sum Operator
class SumOperator : public ReduceOperator {
 public:
  SumOperator(int device_id, c10::ScalarType scalarType)
      : ReduceOperator(
            device_id,
            "reduce_sum_fwd_" +
                habana_helpers::name_suffix_from_type(scalarType)) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};

//
// AnyDimOut Operator
class AnyDimOutOperator : public HabanaOperator {
 public:
  AnyDimOutOperator(int device_id, const std::string& guid)
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
// AnyDim Operator
class AnyDimOperator : public AnyDimOutOperator {
 public:
  AnyDimOperator(int device_id, const std::string& guid)
      : AnyDimOutOperator(device_id, guid) {
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

//
// Any Operator
class AnyOperator : public HabanaOperator {
 public:
  AnyOperator(int device_id, const std::string& guid) : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
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
      bool is_output_persistent = false) override;
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
      bool is_output_persistent = false) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs);
};