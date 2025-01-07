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
namespace habana {
//
// Slice Operator
class SliceOperator : public HabanaOperator {
 public:
  SliceOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("slice") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY,
         LayoutFormat::NCHW,
         LayoutFormat::NCHW,
         LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    this->setNoComputeFlag();
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

  at::Tensor AllocateOutputTensor(
      const at::Tensor& self,
      int64_t& dim,
      int64_t& start,
      int64_t& end,
      int64_t& step,
      const OutputMetaData& output_metadata);

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t& dim,
      int64_t& start,
      int64_t& end,
      int64_t& step);

  static std::vector<std::vector<int64_t>> compute_output_shape(
      const std::vector<int64_t>& self_size,
      int64_t& dim,
      int64_t& start_val,
      int64_t& end_val,
      int64_t& step);

  void ValidateSliceInputs(
      std::vector<int64_t>& inp_shape,
      std::vector<int64_t>& out_shape,
      std::vector<int64_t>& step,
      std::vector<int64_t>& start);

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  static std::vector<int64_t> GetH2DTensorData(
      const at::Tensor& host_tensor,
      bool is_dry_run,
      bool is_min_shape_inference);

  static std::vector<int64_t> ComputeParamsfromH2DTensor(
      const at::Tensor& host_tensor);

  static std::vector<int64_t> get_step_tensor(std::vector<int64_t> h2d_vec);
  static std::vector<int64_t> get_start_tensor(std::vector<int64_t> h2d_vec);

  void UpdateMaxPassSliceInputs(
      std::vector<int64_t>& inp_shape,
      std::vector<int64_t>& out_shape,
      std::vector<int64_t>& step,
      std::vector<int64_t>& start,
      std::vector<int64_t>& min,
      std::vector<int64_t>& max);
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
      const OutputMetaDataVector& output_metadata) override;
};

// Gather Operator
//
class GatherOperator : public HabanaOperator {
 public:
  GatherOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("gather_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;

 private:
  at::Tensor AllocateOutput(
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
};

class GatherElemOperator : public HabanaOperator {
 public:
  GatherElemOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("gather_elements_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t dim_,
      const at::Tensor& index);

 private:
  at::Tensor AllocateOutput(
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
};

// ScatterWrapperOperator Operator
//
class ScatterWrapperOperator : public HabanaOperator {
 public:
  ScatterWrapperOperator(
      int device_id,
      c10::ScalarType scalarType,
      const std::string& guid,
      bool is_inplace = false)
      : HabanaOperator(get_guid_with_precision(guid, scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY, LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    inplace = is_inplace;
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void SetPTOutput(torch::jit::Stack& inputs) override;

  static std::vector<int64_t> compute_output_shape(const at::Tensor& self);

  at::Tensor AllocateOutput(
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
  bool inplace;
};

class ScatterHelperOperator : public ScatterWrapperOperator {
 public:
  ScatterHelperOperator(int device_id, c10::ScalarType scalarType)
      : ScatterWrapperOperator(device_id, scalarType, "scatter_fwd") {}
};

// ScatterAddOperator Operator
// This operator uses UnsortedScatterAddOperator on G2 and above
// It sorts the index tensor and rearranges source tensor
// and uses SortedScatterAddOperator on G1
class ScatterAddOperator : public ScatterWrapperOperator {
 public:
  ScatterAddOperator(int device_id, c10::ScalarType scalarType)
      : ScatterWrapperOperator(device_id, scalarType, "scatter_add_fwd") {}
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// This operator does not expect that index given to it is
// sorted (hence source also does not need re arrangement)
// This is because unsorted_scatter_add_fwd_ guid can work on
// unsorted index tensor. This is used only on G2 and above.
class UnsortedScatterAddOperator : public ScatterWrapperOperator {
 public:
  UnsortedScatterAddOperator(int device_id, c10::ScalarType scalarType)
      : ScatterWrapperOperator(
            device_id,
            scalarType,
            "unsorted_scatter_add_fwd") {}
};
// This operator expects that index and source given to it are such that
// the index is already sorted and the source rearranged accordingly
class SortedScatterAddOperator : public ScatterWrapperOperator {
 public:
  SortedScatterAddOperator(int device_id, c10::ScalarType scalarType)
      : ScatterWrapperOperator(device_id, scalarType, "scatter_add_fwd_") {}
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
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;

 private:
  at::Tensor AllocateOutputTensor(
      const at::Tensor& self,
      int64_t& dim,
      int64_t& index);
};

// ScatterNdONNX operator
class ScatterNdONNXOperator : public HabanaOperator {
 public:
  ScatterNdONNXOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("scatter_nd_onnx_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;

 private:
  bool isInputValid(torch::jit::Stack& inputs);
};

// ScatterND operator
class ScatterNdOperator : public HabanaOperator {
 public:
  ScatterNdOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("scatter_nd_fwd", scalarType)),
        scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

 protected:
  c10::ScalarType scalarType_;
};

// IndexPutOperator
class IndexPutOperator : public HabanaOperator {
 public:
  IndexPutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("index_put_fwd", scalarType)),
        scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
  std::vector<int64_t> broadcast_size(at::TensorList indices);

 private:
  void AllocateAndAddSynapseNodeBoolIndices(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);
  void AllocateAndAddSynapseNodeNonBoolIndices(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);
};

// IndexPutOperator for DS frontend
class IndexPutOperator2 : public HabanaOperator {
 public:
  IndexPutOperator2(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("index_put2_fwd", scalarType)),
        scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
};

// IndexAddOperator
class IndexAddOperator : public HabanaOperator {
 public:
  IndexAddOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("index_add_fwd_filler", scalarType)),
        scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
};

// IndexAddV2Operator
class IndexAddV2Operator : public HabanaOperator {
 public:
  IndexAddV2Operator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("index_add_fwd_filler", scalarType)),
        scalarType_(scalarType) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) final;

 protected:
  c10::ScalarType scalarType_;
};

//
// Arange Operator
class ArangeOperator : public HabanaOperator {
 public:
  ArangeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("range", scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static int GetOutputSize(
      at::Scalar start_,
      at::Scalar end_,
      at::Scalar step_);

  void SetPTOutputs(torch::jit::Stack& inputs) override;
};

//
// Arange Operator
class ArangeOperatorHT : public ArangeOperator {
 public:
  ArangeOperatorHT(int device_id, c10::ScalarType scalarType)
      : ArangeOperator(device_id, scalarType) {}

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// torch::unique Operator
class Unique_Operator : public HabanaOperator {
 public:
  Unique_Operator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("unique_fwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// torch::_unique2 Operator
class UniqueOperator : public HabanaOperator {
 public:
  UniqueOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("unique_fwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void SetPTOutputs(torch::jit::Stack& inputs) override;
};

// torch::unique_dim Operator
class UniqueDimOperator : public HabanaOperator {
 public:
  UniqueDimOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("unique_fwd", scalarType)) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Squeeze operator
class SqueezeOperator : public HabanaOperator {
 public:
  SqueezeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("squeeze") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    this->setNoComputeFlag();
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t dim);

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
};

// Unsqueeze operator
class UnsqueezeOperator : public HabanaOperator {
 public:
  UnsqueezeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("expand_dims") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
    this->setNoComputeFlag();
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      int64_t dim);

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
};

} // namespace habana
