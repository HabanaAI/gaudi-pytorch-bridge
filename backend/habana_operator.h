/*******************************************************************************
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
#include <synapse_api_types.h>
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/habana_types.h"
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_info.h"
#include "backend/helpers/tensor_shape.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/graph.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "habana_helpers/logging.h"
#include "include/habanalabs/hpu_custom_op.h"

#include <ATen/Tensor.h>
#include <c10/util/ArrayRef.h>
#include <torch/csrc/jit/ir/ir.h>

#include <absl/types/any.h>

#include <functional>
#include <iostream>
#include <memory>
#include <string_view>
#include <vector>

using OptionalIntArrayRef = at::OptionalIntArrayRef;

#define DATATYPE_OF_INDEX c10::ScalarType::Long

// This prefix indicates no TPC kernel exists, but op name with
// this prefix is used in eager. Null string avoided, since this
// value is used in eager caching and so per op, unique string
// is required.
const std::string NO_TPC = "[NoTPCKernel]";

// For compound Ops, there is not GUID, so null string used
const std::string NULL_GUID("");

// Set empty size and strides for 0d Tensor
#define SET_SIZE_STRIDE_0D(self)                     \
  self.unsafeGetTensorImpl()->set_sizes_and_strides( \
      at::IntArrayRef{}, at::IntArrayRef{});

// Set size and strides for 1d Tensor
#define SET_SIZE_STRIDE_1D(self)                     \
  self.unsafeGetTensorImpl()->set_sizes_and_strides( \
      at::IntArrayRef{1}, at::IntArrayRef{1});

// Utility Macros to handle 0d tensors input
#define CONVERT_0D_TO_1D(self)                         \
  if (0 == self.dim()) {                               \
    self.unsafeGetTensorImpl()->set_sizes_and_strides( \
        at::IntArrayRef{1}, at::IntArrayRef{1});       \
  }
#define CONVERT_1D_TO_0D(self, out)                    \
  if (0 == self.dim()) {                               \
    self.unsafeGetTensorImpl()->set_sizes_and_strides( \
        at::IntArrayRef{}, at::IntArrayRef{});         \
    out.unsafeGetTensorImpl()->set_sizes_and_strides(  \
        at::IntArrayRef{}, at::IntArrayRef{});         \
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
using PytorchKernelContextPtr = std::unique_ptr<PytorchKernelContext>;
using HabanaOperatorPtr = std::shared_ptr<HabanaOperator>;
using RegisterFunc =
    std::function<HabanaOperatorPtr(const int, c10::ScalarType)>;
using RegisterCustomFunc =
    std::function<HabanaOperatorPtr(const int, std::string)>;

const size_t NO_INPUTS = 0xFFFFFFFF;

enum ShapeTensorType { kShapeTensorNone = 0, kShapeTensor, kDeviceShapeTensor };

struct TensorMetaData {
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  c10::ScalarType dtype;
  c10::MemoryFormat mf;

  TensorMetaData(
      std::vector<int64_t> sz,
      std::vector<int64_t> st,
      c10::MemoryFormat f)
      : sizes(std::move(sz)), strides(std::move(st)), mf(f) {}

  TensorMetaData(
      std::vector<int64_t> sz,
      std::vector<int64_t> st,
      c10::ScalarType type,
      c10::MemoryFormat f)
      : sizes(std::move(sz)), strides(std::move(st)), dtype(type), mf(f) {}
};

// Return value for InferOutputMeta Function
class InferOutputMetaRetType;
using IdxTensorTuple = std::tuple<int32_t, at::Tensor>;
using InferOutputMetaRetTypePtr = std::shared_ptr<InferOutputMetaRetType>;
class InferOutputMetaRetType {
 public:
  InferOutputMetaRetType(bool flag = false) : empty_flag(flag) {}
  bool empty() {
    return empty_flag;
  }
  void set_empty(bool flag = true) {
    empty_flag = flag;
  }
  void AddOutputTensor(const TensorMetaData& data);
  void AddUndefindedOutputTensor() {
    ++num_undefined_outputs;
  }
  void AddIntermediateTensor(const TensorMetaData& data);
  void AddShapeTensor(const TensorMetaData& data);
  void AddDupTensor(const TensorMetaData& data);
  const IdxTensorTuple& GetOutputTensor(size_t index);
  const IdxTensorTuple& GetShapeTensor(size_t index);
  void MoveToOutput(IdxTensorTuple&& data);
  void RemoveOutput(size_t index);
  size_t GetKernelSize() {
    return kernel_outputs.size();
  }
  InferOutputMetaRetType& GetKernel(size_t index) {
    return *kernel_outputs.at(index);
  }
  const std::vector<InferOutputMetaRetTypePtr>& GetKernels() const {
    return kernel_outputs;
  }
  const std::vector<IdxTensorTuple>& GetOutputTensor() const {
    return output_tensors;
  }
  const std::vector<IdxTensorTuple>& GetShapeTensor() const {
    return shape_tensors;
  }
  unsigned GetNumUndefinedOutputTensors() const {
    return num_undefined_outputs;
  }

  InferOutputMetaRetType call_InferOutputMeta(
      HabanaOperatorPtr kernel,
      torch::jit::Stack& inputs);

  const std::vector<InferOutputMetaRetTypePtr>& GetKernelOutputs() const {
    return kernel_outputs;
  }

 private:
  void AddTensor(const TensorMetaData& data, std::vector<IdxTensorTuple>& v);
  std::vector<IdxTensorTuple> output_tensors;
  std::vector<IdxTensorTuple> shape_tensors;
  std::vector<IdxTensorTuple> dup_tensors;
  unsigned num_undefined_outputs = 0;

  std::vector<InferOutputMetaRetTypePtr> kernel_outputs;
  bool empty_flag{false};
};

struct PtInputIdxAndSynHelpTensor {
  int pt_input_idx;
  synapse_helpers::tensor_or_ref sh_t;
  int syn_input_idx;
};

//
// The Pytorch kernel context holds the operator context
// which includes the pytorch tensors, synapse tensor and
// params information for the operator
class PytorchKernelContext {
 public:
  int device_id_;
  std::string node_type_;
  std::vector<at::Tensor> pt_inputs_;
  std::vector<at::Tensor> pt_outputs_;
  std::deque<synapse_helpers::tensor_or_ref> syn_inputs_;
  std::deque<synapse_helpers::tensor_or_ref> syn_outputs_;

  // Normally in inplace and _out kernel variants, input tensors that are being
  // updated inplace are also returned as an output. However, there's at least
  // one case where this rule is not fulfilled - e.g. native_batch_norm updates
  // running_mean and running_var input tensors, but doesn't return them as
  // outputs. It's not possible to treat such tensors as _out or normal inplace
  // tensors, so additional handling is needed.
  std::deque<PtInputIdxAndSynHelpTensor> syn_implicit_outputs_;

  std::set<unsigned int> excluded_output_indices_;
  size_t recipe_key_;

  absl::any params_;
  size_t params_size_;
  bool is_duplicate_input_{false};
  std::deque<synapse_helpers::tensor_or_ref> syn_input_orig_;
  c10::optional<synapse_helpers::tensor_or_ref> syn_seed_;
};

struct KernelMetaData {
  std::vector<LayoutFormat> input_layout;
  std::vector<LayoutFormat> output_layout;
  std::vector<synapse_helpers::layouts::SynapseLayoutFormat>
      synapse_input_layout;
  std::vector<synapse_helpers::layouts::SynapseLayoutFormat>
      synapse_output_layout;
  std::vector<size_t> tpc_input_order;
  bool changes_dims;
  KernelMetaData() : changes_dims(false) {}
};

class OutputMetaData {
 public:
  std::string name{};
  std::string module_name{};
  bool persistent{false};
  bool external{false};
  at::ScalarType dtype{at::ScalarType::Undefined};
  std::vector<int64_t> shape{};
  std::vector<int64_t> strides{};
  at::Layout layout{};
  at::MemoryFormat mem_format{};
  std::optional<at::Tensor> allocated_tensor{};
  bool undefined{false};
  c10::optional<unsigned> exp_bias{c10::nullopt};

  OutputMetaData(const torch::jit::Value& value) : name(value.debugName()){};
  OutputMetaData(
      at::ScalarType dtype,
      std::vector<int64_t> shape,
      std::vector<int64_t> strides,
      at::Layout layout,
      at::MemoryFormat mem_format)
      : dtype(dtype),
        shape(std::move(shape)),
        strides(std::move(strides)),
        layout(layout),
        mem_format(mem_format) {}
  OutputMetaData() = default;
};
using OutputMetaDataVector = std::vector<OutputMetaData>;

// Utility method to select a subset of metadata vector
template <class T>
std::vector<T> SelectVectorIndices(
    const std::vector<T>& src,
    const std::vector<unsigned int>& indices) {
  std::vector<T> result;
  result.reserve(indices.size());
  for (auto index : indices) {
    if (index < src.size())
      result.push_back(src.at(index));
  }
  HABANA_ASSERT(result.size() == indices.size());
  return result;
}

std::string get_guid_with_precision(
    const std::string_view guid,
    c10::ScalarType dtype,
    bool use_int64 = false);

//
// Generic Operator implementation class, holds the operator context
// kernel meta data and helper methods for adding the node to the
// synapse graph and compilation of synapse graph
class HabanaOperator {
 public:
  HabanaOperator(const std::string guid) : guid_(guid) {}

  static bool isFp8Op(const std::string_view guid);

  //
  // Creates graph builder context, based on the device
  void CreateSynContext(int device_id, std::string node_type = "") {
    p_context_ = std::make_unique<PytorchKernelContext>();
    p_context_->device_id_ = device_id;
    p_context_->node_type_ = node_type;
    p_context_->recipe_key_ = 0;
  }

  // Set the op guid - useful on cases where there might have to be
  void SetGuid(std::string guid) {
    guid_ = guid;
  }

  // Get the op guid - Incase guid specific actions needs to be taken
  const std::string& GetGuid() const {
    return guid_;
  }

  void dump(torch::jit::Node* node, const at::Stack& stack);

  // Executes the synapse graph
  virtual void Compile(synapse_helpers::graph& graph);

  virtual void Execute(size_t key);
  virtual void Execute(size_t key, const std::vector<at::Tensor>& inputs);
  virtual void Execute(
      size_t key,
      const std::vector<at::Tensor>& inputs,
      const at::Tensor& output);
  virtual void Execute(
      size_t key,
      const std::vector<at::Tensor>& inputs,
      const std::vector<at::Tensor>& outputs);
  virtual void Execute(
      size_t key,
      const std::vector<at::Tensor>& inputs,
      torch::jit::Stack& output);
  virtual void SetPTInputs(const std::vector<at::Tensor>& inputs);
  virtual void SetPTOutput(const at::Tensor& output);
  virtual void SetPTOutput(torch::jit::Stack& inputs);
  virtual void SetPTOutputs(torch::jit::Stack& inputs);
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

  virtual InferOutputMetaRetType InferOutputMeta(torch::jit::Stack& inputs);

  //
  // Method to add a single tensor to graph builder context, also populates the
  // context params -- this is needed when we need the address of syn_tensor
  // being created
  virtual synapse_helpers::tensor& AllocateSynapseInput(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      bool is_persistent = false,
      synTensorType shape_tensor_type = DATA_TENSOR,
      void* host_ptr = nullptr,
      const std::string& idx = std::string());

  //
  // Method to add a single shape tensor to graph builder context, also
  // populates the context params -- this is needed when we need the address of
  // syn_tensor being created
  virtual synapse_helpers::tensor& AllocateSynapseShapeTensor(
      synapse_helpers::graph& graph,
      const at::Tensor& input,
      synTensorType shape_tensor_type = SHAPE_TENSOR,
      void* host_ptr = nullptr);

  //
  // Method to add a single shape tensor to graph builder context, also
  // populates the context params -- this is needed when we need the address of
  // syn_tensor being created
  virtual synapse_helpers::tensor& AllocateSynapseShapeTensor(
      synapse_helpers::graph& graph,
      const at::IntArrayRef& input_shapes,
      synDeviceId syn_device,
      synTensorType shape_tensor_type = SHAPE_TENSOR,
      void* host_ptr = nullptr);

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
      const OutputMetaData& output_metadata,
      bool is_shape_tensor = false);

  //
  // Method to add output tensors to graph builder context
  // of supported synapse dtype
  virtual void AllocateSynapseOutput(
      synapse_helpers::graph& graph,
      const at::Tensor& output,
      const synDataType synType,
      const OutputMetaData& output_metadata,
      bool is_shape_tensor = false);

  // Method to add output tensors to graph builder context
  virtual void AllocateSynapseInplaceOutput(
      synapse_helpers::graph& graph,
      bool external);

  //
  // Method to add muliple output tensors to graph builder context
  virtual void AllocateSynapseOutputs(
      synapse_helpers::graph& graph,
      const std::vector<at::Tensor>& outputs,
      const OutputMetaDataVector& output_metadata);

  virtual void AllocateAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const OutputMetaDataVector& output_metadata);

  virtual void ReuseMemoryAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
      const OutputMetaDataVector& output_metadata);

  //
  // destructor
  virtual ~HabanaOperator();

  virtual std::vector<at::Tensor>& GetOutputs() const {
    return p_context_->pt_outputs_;
  }

  virtual std::deque<synapse_helpers::tensor_or_ref>& GetSynOutputs() const {
    return p_context_->syn_outputs_;
  }

  virtual std::deque<PtInputIdxAndSynHelpTensor>& GetSynImplicitOutputs()
      const {
    return p_context_->syn_implicit_outputs_;
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
  virtual DMAInputGeneratorType getDMAInputGeneratorType();

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
    op->setDeterministic(deterministic);
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

  void clear_all_pt_and_syn_tensors() {
    p_context_->syn_outputs_.clear();
    p_context_->syn_inputs_.clear();
    p_context_->pt_outputs_.clear();
    p_context_->pt_inputs_.clear();
  }

  static std::vector<int64_t> CalculateStrides(
      const at::IntArrayRef sizes,
      c10::MemoryFormat format);

  virtual synapse_helpers::tensor_or_ref& SynInput(int index) {
    return p_context_->syn_inputs_.at(index);
  }

  void setDeterministic(bool val) {
    deterministic = val;
  }

  bool getDeterministic() {
    return deterministic;
  }

  void setNoComputeFlag() {
    no_compute_flag = true;
  }

  bool getNoComputeFlag() {
    return no_compute_flag;
  }

  synapse_helpers::tensor AllocateConstantSynapseTensor(
      synapse_helpers::graph& graph,
      const c10::Scalar& value);

  synapse_helpers::tensor& AllocateSeed(
      synapse_helpers::graph& graph,
      const at::Tensor& seed_tensor);

  void SetExecutionMode(habana_helpers::HabanaFrontendTypes mode) {
    execution_mode = mode;
  }

  const habana_helpers::HabanaFrontendTypes& GetExecutionMode() {
    return execution_mode;
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
  bool deterministic{false};
  habana_helpers::HabanaFrontendTypes execution_mode{
      habana_helpers::HabanaFrontendTypes::INVALID};
  bool no_compute_flag{false};
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
