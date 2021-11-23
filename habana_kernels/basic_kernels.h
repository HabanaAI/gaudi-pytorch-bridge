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
#include <perf_lib_layer_params.h>
#include "habana_kernels/habana_operator.h"

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
      bool is_output_persistent = false) override;

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
      bool is_output_persistent = false) override;
};

//
// Cast Operator (Lazy mode only)
class CastLazyOperator : public habana::HabanaOperator {
 public:
  CastLazyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("cast_lazy") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    kernel_meta_data_.tpc_input_order = {0};
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;

  /**
   * @brief CastKernel params structure
   */
  ns_CastKernel::Params synapse_cast_params_builder() {
    ns_CastKernel::Params cast_params{};
    cast_params.round_mode = CAST_ROUND_HALF_NE;
    return cast_params;
  }
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
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
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
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};
class DummyOperator : public habana::HabanaOperator {
 public:
  DummyOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("dummy") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
  }
  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
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
      bool is_output_persistent = false) override;
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

class StridedInsertOperator : public habana::HabanaOperator {
 public:
  StridedInsertOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("strided_insert") {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NCHW, habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NCHW});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) override;
};

class StridedInsertClOperator : public StridedInsertOperator {
 public:
  StridedInsertClOperator(int device_id, c10::ScalarType scalarType)
      : StridedInsertOperator(device_id, scalarType) {
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC, habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
  }
};
