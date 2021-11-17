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
#include "habana_helpers/tensor_info.h"
#include "habana_helpers/tensor_shape.h"
#include "habana_helpers/tensor_utils.h"
#include "include/habanalabs/hpu_custom_op.h"
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

// This prefix indicates no TPC kernel exists, but op name with
// this prefix is used in eager. Null string avoided, since this
// value is used in eager caching and so per op, unique string
// is required.
const std::string NO_TPC = "[NoTPCKernel]";

// For compound Ops, there is not GUID, so null string used
const std::string NULL_GUID("");

// Utility Macros to handle 0d tensors input
#define CONVERT_0D_TO_1D(self)                                   \
  if (0 == self.dim()) {                                         \
    self.unsafeGetTensorImpl()->set_sizes_and_strides({1}, {1}); \
  }
#define CONVERT_1D_TO_0D(self, out)                            \
  if (0 == self.dim()) {                                       \
    self.unsafeGetTensorImpl()->set_sizes_and_strides({}, {}); \
    out.unsafeGetTensorImpl()->set_sizes_and_strides({}, {});  \
  }

#define KERNEL_FN_DROP_ARG2(className)                     \
  [](const int device_id, c10::ScalarType node_type) {     \
    static_cast<void>(node_type);                          \
    return std::make_shared<habana::className>(device_id); \
  }

#define KERNEL_FN(className)                                          \
  [](const int device_id, c10::ScalarType node_type) {                \
    return std::make_shared<habana::className>(device_id, node_type); \
  }

#define KERNEL_FN_ARG(className, arg)                                      \
  [](const int device_id, c10::ScalarType node_type) {                     \
    return std::make_shared<habana::className>(device_id, node_type, arg); \
  }

#define KERNEL_FN_GLOBAL(className)                           \
  [](const int device_id, c10::ScalarType node_type) {        \
    return std::make_shared<className>(device_id, node_type); \
  }

namespace habana {

class HabanaOperator;
class PytorchKernelContext;
using PytorchKernelContextPtr = std::shared_ptr<PytorchKernelContext>;
using HabanaOperatorPtr = std::shared_ptr<HabanaOperator>;
using RegisterFunc =
    std::function<HabanaOperatorPtr(const int, c10::ScalarType)>;
using RegisterCustomFunc =
    std::function<HabanaOperatorPtr(const int, std::string)>;

enum class LayoutFormat { NHWC = 0, NCHW = 1, HWCK = 2, ANY = 3, INVALID = 4 };
const size_t NO_INPUTS = 0xFFFFFFFF;

enum ShapeTensorType { kShapeTensorNone = 0, kShapeTensor, kDeviceShapeTensor };
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
  bool is_duplicate_input_{false};
  std::deque<synapse_helpers::tensor_or_ref> syn_input_orig_;
};

typedef struct KernelMetaData {
  std::vector<LayoutFormat> input_layout;
  std::vector<LayoutFormat> output_layout;
  std::vector<size_t> tpc_input_order;
  bool changes_dims;
  KernelMetaData() {
    changes_dims = false;
  }
} KernelMetaData;

class OutputMetaData {
 public:
  std::string name;
  // bool persistent;
  OutputMetaData(const torch::jit::Value& value) : name(value.debugName()) {}
};
using OutputMetaDataVector = std::vector<OutputMetaData>;

// Utility method to select a subset of metadata vector
template <class T>
std::vector<T> SelectVectorIndices(
    const std::vector<T>& src,
    const std::vector<unsigned int> indices) {
  std::vector<T> result;
  result.reserve(indices.size());
  for (auto index : indices) {
    if (index >= 0 && index < src.size())
      result.push_back(src.at(index));
  }
  return result;
}

//
// Generic Operator implementation class, holds the operator context
// kernel meta data and helper methods for adding the node to the
// synapse graph and compilation of synapse graph
class HabanaOperator {
 public:
  HabanaOperator() = delete;
  //
  HabanaOperator(const std::string guid) : guid_(guid) {}

  // Given a target layout, get the permute order.
  // If the tensor is to be sent to device from host
  //    - the target_layout is the one expected inside device
  //    - to_device is true
  // If the tensor is to be sent to host from device
  //    - the target_layout is the one expected in host
  //    - to_device is false
  static const std::array<int64_t, 4>& getPermuteOrder(
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
  virtual void SetPTOutput(torch::jit::Stack& inputs);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
  virtual void SetPTOutputs(std::vector<at::Tensor>& outputs);
  virtual void SetOutputMetadata(int index, const OutputMetaData& md);
  virtual void SetOutputMetadata(const OutputMetaDataVector& md);
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
      bool is_persistent = false,
      synTensorType shape_tensor_type = DATA_TENSOR);

  //
  // Method to add a single shape tensor to graph builder context, also
  // populates the context params -- this is needed when we need the address of
  // syn_tensor being created
  virtual synapse_helpers::tensor& AllocateSynapseShapeTensor(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      synTensorType shape_tensor_type = SHAPE_TENSOR);

  //
  // If Synapse tensor is already exists for the py torch tensor, we just add
  // the synapse tensor to the context
  virtual synapse_helpers::tensor_or_ref& SetSynapseInput(
      synapse_helpers::tensor_or_ref&& tensor);

  //
  // If Synapse tensor is already exists for the py torch tensor, we just add
  // the synapse tensor to the context
  virtual synapse_helpers::tensor_or_ref& SetSynapseInput(
      synapse_helpers::tensor& tensor);

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
      bool is_persistent = false,
      bool is_shape_tensor = false,
      bool use_metadata = true);

  //
  // Method to add output tensors to graph builder context
  // of supported synapse dtype
  virtual void AllocateSynapseOutput(
      synapse_helpers::graph& graph,
      const at::Tensor& output,
      const synDataType synType,
      bool is_persistent = false,
      bool is_shape_tensor = false,
      bool use_metadata = true);

  // Method to add output tensors to graph builder context
  virtual void AllocateSynapseInplaceOutput(synapse_helpers::graph& graph);

  //
  // Method to add muliple output tensors to graph builder context
  virtual void AllocateSynapseOutputs(
      synapse_helpers::graph& graph,
      const std::vector<at::Tensor>& outputs,
      std::vector<bool> is_persistent,
      std::vector<bool> use_metadata);

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      bool is_output_persistent = false);

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      std::vector<bool> is_output_persistent);

  //
  // destructor
  virtual ~HabanaOperator();

  virtual std::vector<at::Tensor>& GetOutputs() const {
    return p_context_->pt_outputs_;
  }

  virtual std::deque<synapse_helpers::tensor_or_ref>& GetSynOutputs() const {
    return p_context_->syn_outputs_;
  }

  virtual std::vector<at::Tensor>& GetInputs() const {
    return p_context_->pt_inputs_;
  }

  virtual std::deque<synapse_helpers::tensor_or_ref>& GetSynInputs() const {
    return p_context_->syn_inputs_;
  }

  virtual std::set<unsigned int>& GetSynOutputIndicesExcludedInNode() const {
    return p_context_->excluded_output_indices_;
  }

  virtual const KernelMetaData& GetKernelMetaData() const {
    return kernel_meta_data_;
  }

  virtual const std::vector<HabanaOperatorPtr> GetKernels() const {
    return kernels_;
  }

  // For populating the inputs that need to be created in host and DMA
  // transferred to the device before the execution of the graph.
  virtual getDMAInputTensorCBType getDMAInputTensorCB();

  // To communicate patching info for tensors which are not part of graph
  virtual std::vector<std::tuple<std::string, at::Tensor, uint64_t>>
  getAppendedTensorInfos();

  virtual const std::vector<std::pair<at::Tensor, at::Tensor>>
  GetDMACandidates() {
    // returns a vector of pairs of tensors
    // first in the pair is 'from' tensor for dma
    // second is the 'target' tensor
    return {};
  }

  template <typename T, typename... Args>
  std::shared_ptr<T> make_operator(Args... args) {
    auto op = std::make_shared<T>(args...);
    kernels_.emplace_back(op);
    return op;
  }

  void set_is_duplicate_input_flag(bool f) {
    p_context_->is_duplicate_input_ = f;
  }

  void add_syn_input_tensor_orig(synapse_helpers::tensor& inp_orig) {
    p_context_->syn_input_orig_.push_back(inp_orig);
  }

  void clear_syn_input_tensor_orig() {
    p_context_->syn_input_orig_.clear();
  }

 protected:
  virtual void AddNodeToSynapseGraph(
      synapse_helpers::graph& graph,
      void* params,
      size_t params_size);

  std::string guid_;
  PytorchKernelContextPtr p_context_;
  KernelMetaData kernel_meta_data_;
  // Store the info on intermediate tensors inserted(not part of graph)
  // THis needs to be communicated to lowering kernel as these additions
  // are invisible there(only graph mappings are queried)
  std::vector<std::tuple<std::string, at::Tensor, uint64_t>>
      appended_tensor_infos;

  //
  std::vector<HabanaOperatorPtr> kernels_;
  std::vector<OutputMetaData>
      output_metadata_; // Must be ordered by allocation order
  unsigned output_allocation_index_ = 0;
};

class RegisterKernel {
 public:
  RegisterKernel& add(const std::string& op, RegisterFunc func) {
    // Construct OperatorName from op
    c10::OperatorName opname = getOperatorName(op);

    TORCH_CHECK(!kernels_.count(opname), opname, " is already registered!");
    kernels_.emplace(opname, func);
    return *this;
  }

  RegisterKernel& add_custom_op(
      const std::string& op,
      RegisterCustomFunc func,
      habana::custom_op::HabanaCustomOpDescriptor desc) {
    c10::OperatorName opname = getOperatorName(op);
    user_cutom_ops_.emplace(opname, func);
    user_cutom_desc_.emplace(opname, desc);
    return *this;
  }

  // Getting user's custom op descriptor from custom op map.
  habana::custom_op::HabanaCustomOpDescriptor& get_custom_op_desc(
      const std::string& op) {
    c10::OperatorName opname = getOperatorName(op);
    return user_cutom_desc_[opname];
  }

  HabanaOperatorPtr get(
      const int device_id,
      const at::OperatorName& opname,
      c10::ScalarType node_type) {
    return kernels_.count(opname) ? kernels_[opname](device_id, node_type)
                                  : user_cutom_ops_.count(opname)
            ? user_cutom_ops_[opname](device_id, opname.name)
            : nullptr;
  }

  RegisterKernel() = default;
  RegisterKernel(const RegisterKernel&) = delete;
  RegisterKernel& operator=(const RegisterKernel&) = delete;

 private:
  c10::OperatorName getOperatorName(const std::string& op) {
    std::istringstream iss{op};
    std::string name, overload_name;
    std::getline(iss, name, '.');
    std::getline(iss, overload_name);

    c10::OperatorName opname{name, overload_name};
    return opname;
  }

 private:
  std::unordered_map<c10::OperatorName, RegisterFunc> kernels_;
  std::unordered_map<c10::OperatorName, RegisterCustomFunc> user_cutom_ops_;
  std::unordered_map<
      c10::OperatorName,
      habana::custom_op::HabanaCustomOpDescriptor>
      user_cutom_desc_;
};

RegisterKernel& KernelRegistry();

}; // namespace habana
