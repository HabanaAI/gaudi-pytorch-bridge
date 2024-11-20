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
#include <perf_lib_layer_params.h>
#include "backend/habana_operator.h"

at::Tensor& copy_hpu_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking,
    synapse_helpers::hpuStream_t hpu_stream);

bool is_pinned_hpu(const at::Tensor& self, at::Device device);
at::Tensor pin_memory_hpu(const at::Tensor& self, at::Device device);

// As Strided Layout
class AsStridedLayoutOperator : public habana::HabanaOperator {
 public:
  AsStridedLayoutOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("dummy") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

//
// MemCopy Operator
class MemCopyOperator : public habana::HabanaOperator {
 public:
  MemCopyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("memcpy") {
    static_cast<void>(scalarType);
    kernel_meta_data_.tpc_input_order = {0};
    this->CreateSynContext(device_id);
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
// Identity Operator
class IdentityOperator : public habana::HabanaOperator {
 public:
  IdentityOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("identity") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    this->setNoComputeFlag();
  }
  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};
class DummyOperator : public habana::HabanaOperator {
 public:
  DummyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("dummy") {
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
// As Strided
class AsStridedOperator : public habana::HabanaOperator {
 public:
  AsStridedOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("dummy") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);

    kernel_meta_data_.input_layout.assign({habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NCHW});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
      compute_output_shape(c10::IntArrayRef, c10::IntArrayRef);
};

class SliceInsertOperator : public habana::HabanaOperator {
 public:
  SliceInsertOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("slice_insert") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;

  void ValidateSliceInsertInputs(
      std::vector<int64_t>& inp_shape,
      std::vector<int64_t>& out_shape,
      std::vector<int64_t>& step,
      std::vector<int64_t>& start);

  void FixSliceParams(
      at::Tensor self,
      int64_t& dim,
      int64_t& start,
      int64_t& end,
      int64_t& step);

  void ComputeParams(
      synSliceParamsV2& params,
      at::Tensor self,
      c10::List<int64_t> paramsList,
      const synapse_helpers::graph& graph);

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void ReuseMemoryAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
      const habana::OutputMetaDataVector& output_metadata) override;

  void UpdateMaxPassSliceInputs(
      std::vector<int64_t>& inp_shape,
      std::vector<int64_t>& out_shape,
      std::vector<int64_t>& step,
      std::vector<int64_t>& start,
      std::vector<int64_t>& min,
      std::vector<int64_t>& max);
};
class SliceScatterOperatorDSUtil : public SliceInsertOperator {
 public:
  SliceScatterOperatorDSUtil(int device_id, c10::ScalarType scalarType)
      : SliceInsertOperator(device_id, scalarType) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};
class SliceScatterOperator : public SliceInsertOperator {
 public:
  SliceScatterOperator(int device_id, c10::ScalarType scalarType)
      : SliceInsertOperator(device_id, scalarType) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

class SelectScatterOperator : public SliceScatterOperator {
 public:
  SelectScatterOperator(int device_id, c10::ScalarType scalarType)
      : SliceScatterOperator(device_id, scalarType) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

class StridedInsertOperator : public habana::HabanaOperator {
 public:
  StridedInsertOperator(int device_id, c10::ScalarType)
      : HabanaOperator("strided_insert") {
    CreateSynContext(device_id);

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NCHW,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NCHW});
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
  void ReuseMemoryAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
      const habana::OutputMetaDataVector& output_metadata) override;
  static void compute_params(
      HabanaOperator&,
      synStridedOpParams&,
      const torch::jit::Stack& inputs,
      synapse_helpers::graph& graph);
  static void compute_params_h2d(
      HabanaOperator&,
      synStridedOpParams&,
      const torch::jit::Stack& inputs,
      synapse_helpers::graph& graph);
  static bool verifyViewMemoryAccess(
      const at::Tensor& real,
      const at::Tensor& view,
      c10::IntArrayRef& strides,
      int64_t& offset);
};

class StridedInsertClOperator : public StridedInsertOperator {
 public:
  StridedInsertClOperator(int device_id, c10::ScalarType scalarType)
      : StridedInsertOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
  }
};

class AsStridedScatterOperator : public StridedInsertOperator {
 public:
  AsStridedScatterOperator(int device_id, c10::ScalarType scalarType)
      : StridedInsertOperator(device_id, scalarType) {}

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
};

// As Strided
class StridedViewOperator : public habana::HabanaOperator {
 public:
  StridedViewOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("strided_view") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NCHW});
  }

  virtual habana::InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;
  void compute_params(
      synStridedOpParams& params,
      torch::jit::Stack& inputs,
      synapse_helpers::graph& graph,
      std::vector<int64_t>& size,
      std::vector<int64_t>& strides,
      int64_t& offset);
  void compute_params_h2d(
      synStridedOpParams& params,
      torch::jit::Stack& inputs,
      synapse_helpers::graph& graph,
      std::vector<int64_t>& size,
      std::vector<int64_t>& strides,
      int64_t& offset);
  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;
  void ReuseMemoryAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
      const habana::OutputMetaDataVector& output_metadata) override;
  static std::tuple<std::vector<int64_t>, std::vector<int64_t>>
  compute_output_shape(const at::Tensor&, c10::IntArrayRef, c10::IntArrayRef);
  bool verifyViewMemoryAccess(
      at::Tensor& real,
      at::Tensor& view,
      c10::IntArrayRef& strides,
      int64_t& offset);
};

// As Strided for channels last
/* The implementation follows the implementation of original Asstrided op with
 *the additional change of setting kernel meta data for NHWC layout to signal
 * the permute pass
 */
class StridedViewClOperator : public StridedViewOperator {
 public:
  StridedViewClOperator(int device_id, c10::ScalarType scalarType)
      : StridedViewOperator(device_id, scalarType) {
    static_cast<void>(scalarType);

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
  }
};
