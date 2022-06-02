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
#include "habana_kernels/index_kernels.h"
namespace habana {

class BatchNormForwardOperator : public habana::HabanaOperator {
 public:
  // NOTE: BatchNormForwardOperator node_type differs for training and eval
  BatchNormForwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("bn_fwd") {
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
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
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

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);

  // virtual std::vector<at::Tensor> preProcessInputs(torch::jit::Stack&
  // inputs);
  virtual void preProcessInputs(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs);
  void generateCacheInputs(torch::jit::Stack& inputs);
  void remove_non_persistent_patching_info();
  virtual void SetPTOutputs(torch::jit::Stack& inputs);

  std::vector<at::Tensor>& GetBNInputs();
  std::vector<at::Tensor>& GetBNOutputs();

  torch::jit::Stack& GetInputstack();

  void SetEagerMode() {
    is_eager_mode = true;
  }

  bool isEagerMode() const {
    return is_eager_mode;
  }

 private:
  void insert_memcopy_op(
      synapse_helpers::graph& graph,
      at::Tensor& src,
      int32_t in_position);
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
  torch::jit::Stack input_stack;
  std::vector<at::Tensor> pre_inputs;
  // This has been added so that the input mean/var synapse tensors are
  // preserved Because we create copies due to in-place restrictions, we replace
  // these as inputs Need to pass along as GC will complain if they are missing
  // in patching info
  std::vector<synapse_helpers::tensor_or_ref> mean_var_temp;
  std::vector<std::pair<at::Tensor, at::Tensor>> dma_candidates;
  bool running_vars_def;

  bool is_eager_mode = false;
};

// Used in lazy mode to avoid the memcopy nodes for RMV
class BatchNormForwardRmvOperator : public habana::HabanaOperator {
 public:
  // Used in training mode
  BatchNormForwardRmvOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("bn_fwd_rmv") {
    this->CreateSynContext(device_id);
    scalarType_ = scalarType;
    // assign layouts for input and output tensors

    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
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
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
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

// Used in lazy mode to avoid the memcopy nodes for RMV
class BatchNormInfOperator : public habana::HabanaOperator {
 public:
  // Used in eval mode
  BatchNormInfOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("bn_fwd_inf") {
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

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
};

class BatchNormBackwardOperator : public habana::HabanaOperator {
 public:
  // NOTE: BatchNormBackwardOperator node_type differs for training and eval
  BatchNormBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator("cud_bn_bwd_ex") {
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
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE});
    kernel_meta_data_.synapse_output_layout.assign(
        {synapse_helpers::layouts::SynapseLayoutFormat::WHCN,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::DONT_CARE,
         synapse_helpers::layouts::SynapseLayoutFormat::WHCN});
    // {input, grad, wt, bias, save_mean, save_ivarstd}
    kernel_meta_data_.tpc_input_order = {1, 0, 2, 7, 5, 6};
    resize_done = false;
    preprocessing_done = false;
  }

  virtual OutputShapeInfRetType ComputeOutputShape(
      torch::jit::Stack& inputs) override;

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);

  void preProcessInputs(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs);

  void generateCacheInputs(torch::jit::Stack& inputs);

  void SetPTOutputs(torch::jit::Stack& inputs);

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

class LayerNormOperator : public habana::HabanaOperator {
 public:
  LayerNormOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "layer_norm_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
  std::tuple<at::Tensor, at::Tensor, at::Tensor> AllocatePTOutputs(
      const at::Tensor& input,
      const at::Tensor& bias,
      const at::Tensor& weight,
      int64_t m,
      std::array<bool, 3> is_persistent);
  static std::vector<std::vector<int64_t>> getOutputSizes(
      const at::Tensor& input,
      int m);
};

class LayerNormBackwardOperator : public habana::HabanaOperator {
 public:
  LayerNormBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "layer_norm_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);
  }

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;
  virtual void SetPTOutputs(torch::jit::Stack& inputs) override;
  std::tuple<at::Tensor, at::Tensor, at::Tensor> AllocatePTOutputs(
      const at::Tensor& input,
      const at::Tensor& weight,
      bool is_persistent);
  static std::vector<std::vector<int64_t>> getOutputSizes(
      const at::Tensor& input,
      const at::Tensor& gamma);
};

// Norm Operator
class NormOperator : public HabanaOperator {
 public:
  NormOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "norm_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
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
      : HabanaOperator(
            "lpnorm_fwd_" + habana_helpers::name_suffix_from_type(scalarType)) {
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
            "frobenius_norm_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
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
      : HabanaOperator(
            "fused_norm_" + habana_helpers::name_suffix_from_type(scalarType)) {
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

class InstanceNormOperator : public habana::HabanaOperator {
 public:
  // Used in training mode
  InstanceNormOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "instance_norm_fwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);

    // assign layouts for input and output tensors
    kernel_meta_data_.input_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.output_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
    kernel_meta_data_.tpc_input_order = {0, 2, 1};

  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  // used to compute output shapes of current mean and var
  static std::vector<int64_t> compute_output_shape(
      at::Tensor input,
      c10::MemoryFormat mf);
};

class InstanceNormBackwardOperator : public habana::HabanaOperator {
 public:
  // Used in training mode
  InstanceNormBackwardOperator(int device_id, c10::ScalarType scalarType)
      : HabanaOperator(
            "instance_norm_bwd_" +
            habana_helpers::name_suffix_from_type(scalarType)) {
    this->CreateSynContext(device_id);

    // assign layouts for input and output tensors
    kernel_meta_data_.input_layout.assign({
        habana::LayoutFormat::NHWC,
        habana::LayoutFormat::NHWC,
        habana::LayoutFormat::ANY,
        habana::LayoutFormat::ANY,
        habana::LayoutFormat::ANY,
    });
    kernel_meta_data_.output_layout.assign(
        {habana::LayoutFormat::NHWC,
         habana::LayoutFormat::ANY,
         habana::LayoutFormat::ANY});
  }

  void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata) override;

  // used to compute output shapes of current mean and var
  static std::vector<int64_t> compute_output_shape(
      at::Tensor input,
      c10::MemoryFormat mf);
};

} // namespace habana
