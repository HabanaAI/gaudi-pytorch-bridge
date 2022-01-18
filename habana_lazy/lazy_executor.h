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
#include <thread>
#include "habana_lazy/hpu_lazy_tensors.h"
#include "habana_lazy/ir.h"
#include "habana_lazy/tensor_impl.h"
#include "synapse_helpers/util.h"
#include "torch/csrc/jit/ir/ir.h"

enum LazyExecutionMode { kLAZY = 0, kLOWERING };

using Graph = torch::jit::Graph;
using GraphPtr = std::shared_ptr<Graph>;

namespace habana_lazy {
enum StrideOPType {
  kStridedOpDefault = 0,
  kStridedOpView,
  kStridedOpSlice,
  kStridedOpSelect,
  kStridedOpTranspose
};

struct StridedOpSliceParams {
  int64_t dim;
  c10::optional<int64_t> start;
  c10::optional<int64_t> end;
  int64_t step;
};

struct StridedOpSelectParams {
  int64_t dim;
  int64_t index;
};

union OpParams {
  StridedOpSliceParams slice_param;
  StridedOpSelectParams select_param;
  OpParams(){};
};

struct StrideParams {
  // storing the tensor helps to retain extend the lifetime of tensor until all
  // the views have expired
  at::Tensor t;
  std::vector<int64_t> sizes;
  std::vector<int64_t> strides;
  int64_t offset;
  StrideOPType optype;
  OpParams params;
};

struct HashFn {
  std::size_t operator()(const std::pair<float, at::ScalarType>& pair) const {
    return std::hash<float>()(pair.first) ^
        std::hash<float>()((float)pair.second);
  }
};

class EqualFn {
 public:
  bool operator()(
      const std::pair<float, at::ScalarType>& a,
      const std::pair<float, at::ScalarType>& b) const {
    return a.first == b.first && a.second == b.second;
  }
};

class HbExecutionContext {
 public:
  HbExecutionContext() = default;
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  const LazyExecutionMode& getExecutionMode();
  void setExecutionMode(LazyExecutionMode mode);
  void MarkTensorsExecuted(bool check_executing = true) {
    for (auto& tensor : m_tensor_execution_status) {
      // Mark all the tensors in executing state as done
      if (!check_executing || tensor.second == kEXECUTING) {
        tensor.second = kEXECUTION_COMPLETE;
      }
    }
  }
  void MarkTensorExecuting(int64_t tensor_id);
  void MarkTensorExecuted(int64_t tensor_id);
  void MarkTensorRegistered(int64_t tensor_id);
  void MarkTensorStatus(int64_t tensor_id, LazyTensorExecutionStatus status);

  LazyTensorExecutionStatus getTensorExecutionStatus(int64_t index);

  std::unordered_map<int64_t, LazyTensorExecutionStatus>&
  getTensorExecutionStatus() {
    return m_tensor_execution_status;
  }

  bool isExecutionInLoweringMode() {
    auto mode = per_thread_execution_mode.find(pthread_self());
    if (mode != std::end(per_thread_execution_mode)) {
      return mode->second == kLOWERING;
    }
    return false;
  }

  void removeRetainedTensor(at::Tensor& tensor) {
    for (auto i = m_retained_tensor_list.begin();
         i != m_retained_tensor_list.end();
         ++i) {
      auto list_tensor_impl = i->unsafeGetTensorImpl();
      if (list_tensor_impl == tensor.unsafeGetTensorImpl()) {
        m_retained_tensor_list.erase(i);
      }
    }
  }

  void saveGraph(GraphPtr p_g) {
    mp_g = p_g;
  }

  GraphPtr getGraph() {
    return mp_g;
  }

  void saveInputsAndOutputs(
      ir::ValueList inputVals,
      ir::ValueList outputVals,
      std::vector<habana_lazy::HbLazyTensor>& tensors,
      const std::vector<int>& indices);

  ir::ValueList& getInputs() {
    return m_input_vals;
  }

  ir::ValueList& getOutputs() {
    return m_output_vals;
  }

  std::vector<habana_lazy::HbLazyTensor> getHbLazyTensors() {
    return m_hblazy_tensors;
  }

  void clear() {
    m_retained_tensor_list.clear();
    if (GET_ENV_FLAG_NEW(PT_HPU_CLEAR_SCALAR_MAP_ON_MARKSTEP)) {
      scalar_to_tensor_map.clear();
    }
    hb_tensors_out_view.clear();
  }

  // We want to retain some tensors for special cases where PT releases them
  // but because we are in lazy mode we actually need them for processing
  // later This should only be used in special cases and released on exit
  // cleanly
  std::vector<at::Tensor> m_retained_tensor_list;

  bool m_is_cached = false;

  std::unordered_map<
      std::pair<float, at::ScalarType>,
      at::Tensor,
      HashFn,
      EqualFn>
      scalar_to_tensor_map;

  // maps tensor id corresponding to as_strided's o/p with its i/p stride params
  std::map<int64_t, StrideParams> view_table;
  // maintains most recent version of the original tensor map
  std::map<int64_t, at::Tensor> orig_tensor_map;

  // view tensors that occurs as graph outputs
  std::vector<habana_lazy::HbLazyTensor> hb_tensors_out_view;

 private:
  // A map between unique lazy tensor ID and execution status
  // Although our execution modes are per thread but tensor status is per
  // device This is because we might be juggling between various threads and
  // we want a common state set for our tensors that are flowing throught the
  // device. Device view seems to be most suited for that
  std::unordered_map<int64_t, LazyTensorExecutionStatus>
      m_tensor_execution_status;

  // LazyExecutionMode : per thread execution mode is maintained
  std::unordered_map<pthread_t, LazyExecutionMode> per_thread_execution_mode;

  GraphPtr mp_g;
  ir::ValueList m_input_vals;
  ir::ValueList m_output_vals;
  std::vector<habana_lazy::HbLazyTensor> m_hblazy_tensors;
};

class HbExecutionContextArena {
 public:
  static HbExecutionContextArena Get();
  HbExecutionContext* getDeviceExecutionContext(
      int device = 0); // TODO remove device from everywhere
  HbExecutionContext* createExecutionContext(int device);
  void removeExecutionContext(int device);
  HbExecutionContextArena() = default;

 private:
  // Keep a map of all the execution contexts in play
  // Right now we support  a single context per device, map maintains ID to
  // context map
  std::unordered_map<int, HbExecutionContext*> m_execution_context_list;
};

// The global object for all contexts, we create contexts out of this per
// device as the execution goes on Create a global object for the arena of
// contexts We will keep them aslive as long as program lives and manage
// device contexts across iterations
extern HbExecutionContextArena habana_lazy_executor;
/*
 * Helper functions to manage the tensor creation based on the execution
 * state. The execution states are following -
 * 1. PyTorch creates a tensor
 *    - Create a tensor with storage, and one without storage.
 *      From the one without storage, create a lazy tensor and from
 *      the lazy tensor point to the internal one with storage.
 *      Retrun the one one without storage.
 * 2. Accumulate ops in IR graph
 *    - Create only storage less tensors and return.
 * 3. Lowering creates a tensor
 *    - Create a tensor with storage and return.
 */
bool allocateTensorWithStorage(int device_index);
bool isDeviceInLoweringMode(int device_index);
}; // namespace habana_lazy
