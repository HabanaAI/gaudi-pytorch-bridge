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
#include <ATen/Tensor.h>
#include <absl/types/any.h>
#include <c10/util/ArrayRef.h>
#include <synapse_api_types.h>
#include <torch/csrc/jit/ir/ir.h>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>
#include "backend/habana_device/HPUDevice.h"
#include "backend/helpers/create_tensor.h"
#include "backend/helpers/dynamic_shape_infer.h"
#include "backend/helpers/habana_types.h"
#include "backend/helpers/layout.h"
#include "backend/helpers/tensor_info.h"
#include "backend/synapse_helpers/device_types.h"
#include "backend/synapse_helpers/graph.h"
#include "backend/synapse_helpers/habana_tensor.h"
#include "backend/synapse_helpers/layout_utils.h"
#include "common/utils.h"
#include "habana_helpers/logging.h"
#include "include/habanalabs/hpu_custom_op.h"
#include "include/habanalabs/hpu_custom_op_pt2.h"

using OptionalIntArrayRef = at::OptionalIntArrayRef;

#define DATATYPE_OF_INDEX c10::ScalarType::Long

// This prefix indicates no TPC kernel exists, but op name with
// this prefix is used in eager. Null string avoided, since this
// value is used in eager caching and so per op, unique string
// is required.
const std::string NO_TPC = "[NoTPCKernel]";

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

class InferNodeParams {
 public:
  InferNodeParams(void* data, const size_t size) {
    if (data && size) {
      paramsVec.resize(size);
      const auto cData = reinterpret_cast<uint8_t*>(data);
      std::copy(cData, cData + size, paramsVec.data());
    }
  }
  const void* get_data() const {
    if (paramsVec.empty()) {
      return nullptr;
    }
    return paramsVec.data();
  }
  unsigned get_size() const {
    return paramsVec.size();
  }

 private:
  std::vector<uint8_t> paramsVec;
};

// Return value for InferOutputMeta Function
class InferOutputMetaRetType;
using IdxTensorTuple = std::tuple<int32_t, at::Tensor>;
using InferOutputMetaRetTypePtr = std::shared_ptr<InferOutputMetaRetType>;
class InferOutputMetaRetType {
 public:
  explicit InferOutputMetaRetType(bool flag = false) : empty_flag_(flag) {}

  InferOutputMetaRetType(const InferOutputMetaRetType&) = default;
  InferOutputMetaRetType& operator=(const InferOutputMetaRetType&) = default;
  InferOutputMetaRetType(InferOutputMetaRetType&&) = default;
  InferOutputMetaRetType& operator=(InferOutputMetaRetType&&) = default;
  ~InferOutputMetaRetType() = default;

  bool empty() const {
    return empty_flag_;
  }

  void set_empty(const bool flag = true) {
    empty_flag_ = flag;
  }

  void AddOutputTensor(const TensorMetaData& data);

  void AddUndefinedOutputTensor() {
    ++num_undefined_outputs_;
  }

  void AddIntermediateTensor(const TensorMetaData& data);

  void AddShapeTensor(const TensorMetaData& data);

  void AddDupTensor(const TensorMetaData& data);

  const IdxTensorTuple& GetOutputTensor(size_t index) const;

  const IdxTensorTuple& GetShapeTensor(size_t index) const;

  void MoveToOutput(IdxTensorTuple&& data);

  void RemoveOutput(size_t index);

  void PushOutputTensorAtFront(IdxTensorTuple output_tensor);

  size_t GetKernelSize() const {
    return kernel_outputs_.size();
  }
  void AddNodeParams(void* data, size_t size) {
    node_params_.emplace_back(InferNodeParams(data, size));
  }
  const std::vector<InferNodeParams>& GetNodeParams() const {
    return node_params_;
  }

  InferOutputMetaRetType& GetKernel(size_t index) const {
    return *kernel_outputs_.at(index);
  }

  const std::vector<InferOutputMetaRetTypePtr>& GetKernels() const {
    return kernel_outputs_;
  }

  const std::vector<IdxTensorTuple>& GetOutputTensor() const {
    return output_tensors_;
  }

  const std::vector<IdxTensorTuple>& GetShapeTensor() const {
    return shape_tensors_;
  }

  unsigned GetNumUndefinedOutputTensors() const {
    return num_undefined_outputs_;
  }

  InferOutputMetaRetType& call_InferOutputMeta(
      HabanaOperatorPtr kernel,
      torch::jit::Stack& inputs);

  const std::vector<InferOutputMetaRetTypePtr>& GetKernelOutputs() const {
    return kernel_outputs_;
  }

 private:
  void AddTensor(const TensorMetaData& data, std::vector<IdxTensorTuple>& v);

  std::vector<IdxTensorTuple> output_tensors_;

  std::vector<IdxTensorTuple> shape_tensors_;

  std::vector<IdxTensorTuple> dup_tensors_;

  std::vector<InferNodeParams> node_params_;

  unsigned num_undefined_outputs_{0};

  std::vector<InferOutputMetaRetTypePtr> kernel_outputs_;

  bool empty_flag_{false};
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
  bool changes_dims{};
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
  OutputMetaData(at::ScalarType dtype, std::vector<int64_t> shape)
      : dtype(dtype), shape(std::move(shape)) {}
  OutputMetaData() = default;
};
using OutputMetaDataVector = std::vector<OutputMetaData>;

using SharedMetaTensor = std::pair<int, at::ScalarType>;
using SharedMetaVector = std::vector<SharedMetaTensor>;

inline SharedMetaTensor createOptionalNotPresentSharedMetaTensor() {
  return {0, at::ScalarType::Undefined};
}

struct SharedMetaData {
  struct SharedMetaValidationOptions {
    bool allowLongType = false;
  };
  std::string guid;
  SharedMetaVector inputs_data;
  SharedMetaVector outputs_data;
  SharedMetaValidationOptions options;

  SharedMetaData(const std::string& guid) : guid(guid) {}
  SharedMetaData() = default;
};
using SharedMetaDataVector = std::vector<SharedMetaData>;

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

  virtual bool STMeta(
      habana_helpers::IShapeList& inputs,
      habana_helpers::IShapeList& outputs);

  virtual void ReuseMemoryAndAddSynapseNode(
      synapse_helpers::graph& graph,
      torch::jit::Stack& inputs,
      const std::vector<synapse_helpers::tensor_or_ref>& syn_t_vec,
      const OutputMetaDataVector& output_metadata);

  //
  // destructor
  virtual ~HabanaOperator();

  virtual std::vector<at::Tensor>& GetOutputs() {
    return p_context_->pt_outputs_;
  }

  virtual const std::vector<at::Tensor>& GetOutputs() const {
    return p_context_->pt_outputs_;
  }

  virtual std::deque<synapse_helpers::tensor_or_ref>& GetSynOutputs() {
    return p_context_->syn_outputs_;
  }

  virtual const std::deque<synapse_helpers::tensor_or_ref>& GetSynOutputs()
      const {
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

  virtual synapse_helpers::tensor_or_ref& SynInput(size_t index) {
    return p_context_->syn_inputs_.at(index);
  }

  void setDeterministic(bool val) {
    deterministic = val;
  }

  bool getDeterministic() {
    return deterministic;
  }

  void setContextHints(const std::string& hints_str) {
    hints_str_ = hints_str;
  }

  std::string getContextHints() {
    return hints_str_;
  }

  void setNoComputeFlag() {
    no_compute_flag = true;
  }

  bool getNoComputeFlag() {
    return no_compute_flag;
  }

  synapse_helpers::tensor AllocateConstantSynapseTensor(
      synapse_helpers::graph& graph,
      const c10::Scalar& value,
      c10::optional<at::ScalarType> force_type = c10::nullopt);

  template <class T, class U>
  static void CopyVecToHostPtr(const std::vector<T>& vec, void* host_ptr) {
    if constexpr (std::is_same<T, U>::value) {
      std::copy(vec.begin(), vec.end(), static_cast<T*>(host_ptr));
    } else {
      std::vector<U> vec_temp(vec.size());
      std::transform(vec.begin(), vec.end(), vec_temp.begin(), [](const T& v) {
        return static_cast<U>(v);
      });
      std::copy(vec_temp.begin(), vec_temp.end(), static_cast<U*>(host_ptr));
    }
  }

  // Allocate constant synapse tensor for handling vectors
  template <class T>
  synapse_helpers::tensor AllocateConstantSynapseTensor(
      synapse_helpers::graph& graph,
      int device_id,
      const std::vector<T>& vec,
      at::OptionalIntArrayRef sizes) {
    auto& device = habana::HPUDeviceContext::get_device(device_id);

    void* host_ptr{nullptr};
    at::ScalarType vec_type{};

    const bool is_int64_support_enabled = common::IsInt64Supported();

    constexpr bool is_long = std::is_same<T, int64_t>::value;
    constexpr bool is_double = std::is_same<T, double>::value;

    if constexpr (is_long) {
      vec_type =
          is_int64_support_enabled ? at::ScalarType::Long : at::ScalarType::Int;
    } else if constexpr (is_double) {
      vec_type = at::ScalarType::Float;
    } else {
      vec_type = c10::CppTypeToScalarType<T>::value;
    }

    const auto& host_ptr_size = elementSize(vec_type) * vec.size();
    auto status = device.get_host_memory().malloc(&host_ptr, host_ptr_size);
    HABANA_ASSERT(status == synSuccess, Logger::synStatusToStr(status));

    if (is_long and not is_int64_support_enabled) {
      CopyVecToHostPtr<T, int>(vec, host_ptr);
    } else if (is_double) {
      CopyVecToHostPtr<T, float>(vec, host_ptr);
    } else {
      std::copy(vec.begin(), vec.end(), static_cast<T*>(host_ptr));
    }

    auto const_syn_tensor = habana_helpers::create_const_tensor(
        sizes ? *sizes : at::IntArrayRef{static_cast<int64_t>(vec.size())},
        {1},
        graph,
        false,
        device_id,
        vec_type,
        host_ptr,
        host_ptr_size);

    // Free host_ptr here only since copy_buffer is set true for const tensor
    device.get_host_memory().free(host_ptr);

    // Increment count for const tensors created for scalars
    graph.increment_const_tensors();

    return const_syn_tensor;
  }

  synapse_helpers::tensor& AllocateSeed(
      synapse_helpers::graph& graph,
      const at::Tensor& seed_tensor);

  void SetExecutionMode(habana_helpers::HabanaFrontendTypes mode) {
    execution_mode = mode;
  }

  void SetOpDynamicity(bool is_op_dynamic_) {
    is_op_dynamic = is_op_dynamic_;
  }

  bool GetOpDynamicity() {
    return is_op_dynamic;
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
  std::string hints_str_{};
  bool is_op_dynamic{true};
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

  RegisterKernel& add_legacy_user_custom_op(
      const std::string& op,
      RegisterCustomFunc func,
      habana::custom_op::HabanaCustomOpDescriptor desc) {
    c10::OperatorName opname = getOperatorName(op);
    legacy_user_custom_ops.emplace(opname, func);
    legacy_user_custom_desc_.emplace(opname, desc);
    return *this;
  }

  RegisterKernel& add_user_custom_op(
      RegisterCustomFunc func,
      const habana::custom_op::UserCustomOpDescriptor& desc) {
    c10::OperatorName opname = getOperatorName(desc.getSchemaName());
    user_custom_ops_.emplace(opname, func);
    user_custom_desc_.emplace(opname, desc);
    return *this;
  }

  // Getting user's custom op descriptor from custom op map.
  habana::custom_op::HabanaCustomOpDescriptor& get_legacy_user_custom_op_desc(
      const std::string& op) {
    c10::OperatorName opname = getOperatorName(op);
    return legacy_user_custom_desc_[opname];
  }

  // Getting user's custom op descriptor from custom op map.
  habana::custom_op::UserCustomOpDescriptor& get_user_custom_op_desc(
      const std::string& op) {
    c10::OperatorName opname = getOperatorName(op);
    return user_custom_desc_[opname];
  }

  HabanaOperatorPtr get(
      const int device_id,
      const at::OperatorName& opname,
      c10::ScalarType node_type) {
    if (kernels_.count(opname)) {
      return kernels_[opname](device_id, node_type);
    } else if (user_custom_ops_.count(opname)) {
      return user_custom_ops_[opname](device_id, opname.name);
    } else if (legacy_user_custom_ops.count(opname)) {
      return legacy_user_custom_ops[opname](device_id, opname.name);
    }
    return nullptr;
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
  std::unordered_map<c10::OperatorName, RegisterCustomFunc>
      legacy_user_custom_ops;
  std::unordered_map<
      c10::OperatorName,
      habana::custom_op::HabanaCustomOpDescriptor>
      legacy_user_custom_desc_;
  std::unordered_map<c10::OperatorName, RegisterCustomFunc> user_custom_ops_;
  std::unordered_map<
      c10::OperatorName,
      habana::custom_op::UserCustomOpDescriptor>
      user_custom_desc_;
};

RegisterKernel& KernelRegistry();

}; // namespace habana
