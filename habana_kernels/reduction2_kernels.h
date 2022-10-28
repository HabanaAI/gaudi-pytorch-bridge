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
 * Base class for reduce operators such as Max, Min etc.
 * Should never be instantiated directly.
 */
class Reduce2Operator : public HabanaOperator {
 public:
  Reduce2Operator(int device_id, const std::string& guid)
      : HabanaOperator(guid) {
    this->CreateSynContext(device_id);
  }

  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs);

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class MaxDimOperator : public HabanaOperator {
 public:
  MaxDimOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_max_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  OutputShapeInfRetType ComputeOutputShape(torch::jit::Stack& inputs);

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t dim,
      bool keepdim) {
    auto shape_out = self.sizes().vec();
    dim = at::maybe_wrap_dim(dim, self.dim());
    if (keepdim == true) {
      shape_out[dim] = 1;
    } else {
      shape_out.erase(shape_out.begin() + dim);
    }
    return shape_out;
  };
};

class MaxOperator : public HabanaOperator {
 public:
  MaxOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_max_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape() {
    std::vector<int64_t> shape_out{};
    return shape_out;
  };

 private:
  void ReduceSingle(
      synapse_helpers::graph& graph,
      at::Tensor& input,
      int64_t i,
      synapse_helpers::tensor& syn_input);

  std::vector<Reduce2Operator> ReduceOpList;
};

class MinOperator : public HabanaOperator {
 public:
  MinOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "reduce_min_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape() {
    std::vector<int64_t> shape_out{1};
    return shape_out;
  };

 private:
  void ReduceSingle(
      synapse_helpers::graph& graph,
      at::Tensor& input,
      int64_t i,
      synapse_helpers::tensor& syn_input);

  std::vector<Reduce2Operator> ReduceOpList;
};
} // namespace habana
