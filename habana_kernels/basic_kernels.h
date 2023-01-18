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
#include <perf_lib_layer_params.h>
#include "habana_kernels/habana_operator.h"

at::Tensor& copy_hpu_(
    at::Tensor& self,
    const at::Tensor& src,
    bool non_blocking);

bool is_pinned_hpu(const at::Tensor& self, c10::optional<at::Device> device);
at::Tensor pin_memory_hpu(
    const at::Tensor& self,
    c10::optional<at::Device> device);

//
// Function to adjust and set the correct memory format
// for pytorch tensor
void adjustPTSizes(at::Tensor& t);

//
// Function to check if the tensor is channels last format
bool copy_transpose_valid(const at::Tensor& self, const at::Tensor& src);

//
// ToDtype Operator
class ToDtypeOperator : public habana::HabanaOperator {
 public:
  ToDtypeOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("to_dtype") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const habana::OutputMetaDataVector& output_metadata) override;

  // virtual void SetPTOutput(torch::jit::Stack& inputs) override;
};

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
  }
  virtual habana::OutputShapeInfRetType ComputeOutputShape(
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
  }
  virtual habana::OutputShapeInfRetType ComputeOutputShape(
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
  virtual habana::OutputShapeInfRetType ComputeOutputShape(
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
  compute_output_shape(const at::Tensor&, c10::IntArrayRef, c10::IntArrayRef);
};

// As Strided for channels last
/* The implementation follows the implementation of original Asstrided op with
 *the additional change of setting kernel meta data for NHWC layout to signal
 * the permute pass
 */
class AsStridedClOperator : public AsStridedOperator {
 public:
  AsStridedClOperator(int device_id, c10::ScalarType scalarType)
      : AsStridedOperator(device_id, scalarType) {
    static_cast<void>(scalarType);

    kernel_meta_data_.input_layout.assign({habana::LayoutFormat::NHWC});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
  }
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

  void FixSliceParams(
      at::Tensor self,
      int64_t& dim,
      int64_t& start,
      int64_t& end,
      int64_t& step);

  void ComputeParams(
      synSliceParamsNDims& params,
      at::Tensor self,
      c10::List<int64_t> paramsList,
      const synapse_helpers::graph& graph);

  virtual habana::OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  void ReuseMemoryAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
      const habana::OutputMetaDataVector& output_metadata) override;
};

class StridedInsertOperator : public habana::HabanaOperator {
 public:
  StridedInsertOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("strided_insert") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NCHW,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::NCHW,
         habana::LayoutFormat::NCHW});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NCHW});
  }

  virtual habana::OutputShapeInfRetType ComputeOutputShape(
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
  void compute_params(
      synStridedOpParams&,
      torch::jit::Stack& inputs,
      synapse_helpers::graph& graph);
  void compute_params_h2d(
      synStridedOpParams&,
      torch::jit::Stack& inputs,
      synapse_helpers::graph& graph);
  bool verifyViewMemoryAccess(
      at::Tensor& real,
      at::Tensor& view,
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

  virtual habana::OutputShapeInfRetType ComputeOutputShape(
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
