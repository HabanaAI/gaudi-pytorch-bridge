/**
* Copyright (c) 2021-2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#pragma once

#include "backend/habana_operator.h"
#include "habana_kernels/tensor_shape_kernels.h"

namespace habana {

class MMOperator : public HabanaOperator {
 public:
  MMOperator(int device_id) : HabanaOperator("gemm") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      at::Tensor self,
      at::Tensor other,
      bool self_transposed = false,
      bool other_transposed = false);
};

class BmmOutOperator : public HabanaOperator {
 public:
  BmmOutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("batch_gemm") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class BmmOperator : public BmmOutOperator {
 public:
  BmmOperator(int device_id, c10::ScalarType scalarType)
      : BmmOutOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      const at::Tensor& mat2,
      bool mat1_transposed = false,
      bool mat2_transposed = false);
};

//
// Mv Operator
class MvOperator : public HabanaOperator {
 public:
  MvOperator(int device_id) : HabanaOperator("mv") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

//
// Dot Operator
class DotOperator : public HabanaOperator {
 public:
  DotOperator(int device_id) : HabanaOperator("dot") {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class MatMulOperator : public HabanaOperator {
 public:
  MatMulOperator(int device_id) : HabanaOperator("matmul") {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      const at::Tensor& mat2,
      bool other_transposed = false);
};

class MatmulBackwardOperator : public HabanaOperator {
 public:
  MatmulBackwardOperator(int device_id) : HabanaOperator("matmul_backward") {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

 private:
  void MatBwReshape(
      synapse_helpers::graph& graph,
      at::Tensor& mat,
      std::vector<int64_t> sizes,
      synapse_helpers::tensor& syn_input);

  bool is_specialfold_without_reshape_case(
      int64_t dim1,
      int64_t dim2,
      int64_t dim_out) {
    return (dim_out == 2 and dim1 == dim2 and dim1 >= 3);
  }

  std::vector<HabanaOperatorPtr> ReshapeOpList;
};

class MatMulBwdOperator : public HabanaOperator {
 public:
  MatMulBwdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("matmul_bwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

} // namespace habana
