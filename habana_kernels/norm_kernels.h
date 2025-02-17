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
#include "habana_kernels/index_kernels.h"
namespace habana {

class BNFwdTPCRetIndex {
 public:
  constexpr static char Output = 0;
  constexpr static char SavedMean = 1;
  constexpr static char SavedIStd = 2;
  constexpr static char RunningMean = 3;
  constexpr static char RunningVar = 3;
};

class BNBwdTPCRetIndex {
 public:
  constexpr static char Output = 0;
  constexpr static char BiasGrad = 1;
  constexpr static char WeightGrad = 2;
};

class BatchNormForwardOperator : public habana::HabanaOperator {
 public:
  // Used in training mode
  BatchNormForwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("batch_norm_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
    // assign layouts for input and output tensors

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  }

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void preProcessInputs(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs);

 private:
  at::Tensor create_or_return_tensor_bn(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      uint size,
      at::Device device,
      int syn_index);
  at::Tensor create_or_return_pt_tensor_bn(
      const at::Tensor& input,
      uint size,
      at::Device device);

  c10::ScalarType scalarType_;
  std::vector<synapse_helpers::tensor_or_ref> tensors_;
  std::vector<at::Tensor> pt_inputs;
  std::vector<at::Tensor> pt_outputs;
  std::vector<at::Tensor> pre_inputs;
};

class BatchNormBackwardOperator : public habana::HabanaOperator {
 public:
  // NOTE: BatchNormBackwardOperator node_type differs for training and eval
  BatchNormBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("batch_norm_bwd", scalarType)) {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
    // assign layouts for input and output tensors
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    // {input, grad, mean, istd, weight}
    resize_done = false;
    preprocessing_done = false;
  }

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  void preProcessInputs(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs);

  torch::jit::Stack& GetInputstack() {
    return input_stack;
  };

  bool CheckResizeDone() {
    return resize_done;
  };

  void SetResizeDone() {
    resize_done = true;
  };

  bool CheckProprocessingDone() {
    return preprocessing_done;
  }

  void SetProprocessingDone() {
    preprocessing_done = true;
  };

 private:
  void create_opt_input_tensor_bn_bwd(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      uint size,
      at::Device device,
      int syn_index);

  c10::ScalarType scalarType_;
  std::vector<at::Tensor> pt_inputs;
  torch::jit::Stack input_stack;
  std::vector<at::Tensor> pre_inputs;
  bool resize_done;
  bool preprocessing_done;
};

class BatchNormInfOperator : public habana::HabanaOperator {
 public:
  // Used in eval mode
  BatchNormInfOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("batch_norm_inf", scalarType)) {
    static_cast<void>(scalarType);
    this->CreateSynContext(device_id);
    // assign layouts for input and output tensors

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({habana::LayoutFormat::NHWC});
    kernel_meta_data_.synapse_input_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
  }

  virtual InferOutputMetaRetType InferOutputMeta(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// Norm Operator
class NormOperator : public HabanaOperator {
 public:
  NormOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("norm_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
  static std::vector<int64_t> compute_output_shape(
      const at::Tensor& self,
      at::IntArrayRef dim,
      bool keepdim);

 private:
  void AddL0NormNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
  void AddLInfNormNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaData& output_metadata);
};

// LpNorm Operator
class LpNormOperator : public HabanaOperator {
 public:
  LpNormOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("lpnorm_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {LayoutFormat::ANY, LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// LpNormFrobenius Operator
class LpNormFrobeniusOperator : public HabanaOperator {
 public:
  LpNormFrobeniusOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            get_guid_with_precision("frobenius_norm_fwd", scalarType)) {
    this->CreateSynContext(device_id);
    kernel_meta_data_.input_layout.assign({LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign({LayoutFormat::ANY});
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

// FusedNorm Operator
class FusedNormOperator : public HabanaOperator {
 public:
  FusedNormOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(get_guid_with_precision("fused_norm", scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  // the common portion of code between fused_norm and fused_norm_lazy
  std::shared_ptr<SliceOperator> compute_clip_coeff(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);
};

// FusedNormLazy Operator
// The difference  between FusedNormOperator is out of place implementation of
// gradient clipping This is possible because we can attach the clipped grad
// back to the original tensor (similar to BN RMV) This approach avoids creation
// of duplicate persistent tensors in multinode scenario where
// grad = Strided_view(buckettensor) here strided view will be out of place. so
// we would be adding strided insert nodes after fused norm to get updated
// version of grad
class FusedNormLazyOperator : public FusedNormOperator {
 public:
  FusedNormLazyOperator(int device_id, c10::ScalarType scalarType)
      : FusedNormOperator(device_id, scalarType) {
    this->CreateSynContext(device_id);
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};
} // namespace habana
