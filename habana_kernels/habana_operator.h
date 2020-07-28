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
#include <synapse_api_types.h>
#include "habana_helpers/logging.h"
#include "habana_helpers/tensor_utils.h"
#include "synapse_helpers/device_types.h"
#include "synapse_helpers/graph.h"
#include "synapse_helpers/habana_tensor.h"

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>

#include <absl/types/any.h>

#include <functional>
#include <iostream>
#include <memory>

namespace habana {
enum class LayoutFormat { NHWC = 0, NCHW = 1, HWCK = 2, ANY = 3, INVALID = 4 };

//
// The Pytorch kernel context holds the operator context
// whcih includes the pytorch tensors, synapse tensor and
// params information for the operator
class PytorchKernelContext {
 public:
  int device_id_;
  std::vector<at::Tensor> pt_inputs_;
  std::vector<at::Tensor> pt_outputs_;
  std::deque<synapse_helpers::tensor_or_ref> syn_inputs_;
  std::deque<synapse_helpers::tensor_or_ref> syn_outputs_;
  std::set<unsigned int> excluded_output_indices_;
  size_t recipe_key_;

  absl::any params_;
  size_t params_size_;
};

typedef struct {
  std::vector<LayoutFormat> input_layout;
  std::vector<LayoutFormat> output_layout;
  std::set<int>             valid_input_idx;
} KernelMetaData;

using PytorchKernelContextPtr = std::shared_ptr<PytorchKernelContext>;

//
// Generic Operator implementation class, holds the operator context
// kernel meta data and helper methods for adding the node to the
// synapse graph and compilation of synapse graph
class HabanaOperator {
 public:
  //
  HabanaOperator(const std::string guid) : guid_(guid) {}

  //
  // Given a target layout, get the permute order.
  // If the tensor is to be sent to device from host
  //    - the target_layout is the one expected inside device
  //    - to_device is true
  // If the tensor is to be sent to host from device
  //    - the target_layout is the one expected in host
  //    - to_device is false
  static const at::IntArrayRef& getPermuteOrder(
      const LayoutFormat target_layout,
      bool to_device = true);

  //
  // Creates graph builder context, based on the device
  void CreateSynContext(int device_id) {
    p_context_ = std::make_shared<PytorchKernelContext>();
    p_context_->device_id_ = device_id;
    p_context_->recipe_key_ = 0;
  }

  // Set the op guid - useful on cases where there might have to be
  void SetGuid(std::string guid) {
    guid_ = guid;
  }
  //
  // Executes the synapse graph
  virtual void Compile(synapse_helpers::graph& graph);

  virtual void Execute(size_t key);
  virtual void SetPTInputs(const std::vector<at::Tensor>& inputs);
  virtual void SetPTOutput(const at::Tensor& output);
  virtual void SetPTOutputs(const std::vector<at::Tensor>& outputs);
  virtual size_t GetRecipeKey(
      std::string node,
      std::vector<c10::IValue> stack,
      bool inPlaceOp = false,
      bool outOp = false);
  //
  // Method to add tensors to graph builder context, also populates the context
  // params
  virtual void AllocateSynapseInputs(
      synapse_helpers::graph& graph,
      const std::vector<at::Tensor>& inputs,
      bool is_persistent = false);

  //
  // Method to add a single tensor to graph builder context, also populates the
  // context params -- this is needed when we need the address of syn_tensor
  // being created
  virtual synapse_helpers::tensor& AllocateSynapseInput(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      bool is_persistent = false);

  //
  // If Synapse tensor is already exists for the py torch tensor, we just add
  // the synapse tensor to the context
  virtual synapse_helpers::tensor_or_ref& SetSynapseInput(
      synapse_helpers::tensor_or_ref&& tensor);

  //
  // If Synapse tensor is already exists for the py torch tensor, we just add
  // the synapse tensor to the context
  virtual synapse_helpers::tensor_or_ref& SetSynapseOutput(
      synapse_helpers::tensor_or_ref&& tensor);

  //
  // Method to add output tensors to graph builder context
  virtual void AllocateSynapseOutput(
      synapse_helpers::graph& graph,
      const at::Tensor& output,
      bool is_persistent = false);

  // Method to add output tensors to graph builder context
  virtual void AllocateSynapseInplaceOutput(synapse_helpers::graph& graph);

  //
  // Method to add muliple output tensors to graph builder context
  virtual void AllocateSynapseOutputs(
      synapse_helpers::graph& graph,
      const std::vector<at::Tensor>& outputs,
      bool is_persistent);

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false) = 0;

  //
  // destructor
  virtual ~HabanaOperator();

  virtual std::vector<at::Tensor>& GetOutputs() const {
    return p_context_->pt_outputs_;
  }

  virtual std::deque<synapse_helpers::tensor_or_ref>& GetSynOutputs() const {
    return p_context_->syn_outputs_;
  }

  virtual std::set<unsigned int>& GetSynOutputIndicesExcludedInNode() const {
    return p_context_->excluded_output_indices_;
  }

  virtual const KernelMetaData& GetKernelMetaData() const {
    return kernel_meta_data_;
  }

  //To communicate patching info for tensors which are not part of graph
  virtual std::vector<std::pair<std::string, at::Tensor>> getAppendedTensorInfos();

  virtual const std::vector<std::pair<at::Tensor, at::Tensor>>
  GetDMACandidates() {
    // returns a vector of pairs of tensors
    // first in the pair is 'from' tensor for dma
    // second is the 'target' tensor
    return {};
  }

 protected:
  virtual void AddNodeToSynapseGraph(
      synapse_helpers::graph& graph,
      void* params,
      size_t params_size);

  std::string guid_;
  PytorchKernelContextPtr p_context_;
  KernelMetaData kernel_meta_data_;
  //Store the info on intermediate tensors inserted(not part of graph)
  //THis needs to be communicated to lowering kernel as these additions
  //are invisible there(only graph mappings are queried)
  std::vector<std::pair<std::string, at::Tensor>> appended_tensor_infos;

};

using HabanaOperatorPtr = std::shared_ptr<HabanaOperator>;
using RegisterFunc =
    std::function<HabanaOperatorPtr(const int, c10::ScalarType)>;

class RegisterKernel {
 public:
  RegisterKernel& add(const std::string& node_name, RegisterFunc func) {
    TORCH_CHECK(
        !kernels_.count(node_name),
        "Kernel ",
        node_name,
        " is already registered");
    kernels_.emplace(node_name, func);
    return *this;
  }

  HabanaOperatorPtr get(
      const int device_id,
      const std::string& node_name,
      c10::ScalarType node_type) {
    TORCH_CHECK(
        kernels_.count(node_name), "Kernel ", node_name, " is not registered");
    return kernels_[node_name](device_id, node_type);
  }

  RegisterKernel() = default;
  RegisterKernel(const RegisterKernel&) = delete;
  RegisterKernel& operator=(const RegisterKernel&) = delete;

 private:
  std::map<const std::string, RegisterFunc> kernels_;
};

RegisterKernel& KernelRegistry();

}; // namespace habana
