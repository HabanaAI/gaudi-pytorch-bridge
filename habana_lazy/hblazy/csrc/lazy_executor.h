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
#include "habana_lazy/tensor_impl.h"
#include "synapse_helpers/util.h"

enum LazyExecutionMode { kLAZY = 0, kLOWERING };

namespace habana_lazy {
class HbExecutionContext {
 public:
  HbExecutionContext() = default;
  void RegisterTensor(std::shared_ptr<Data> data);
  void UnregisterTensor(Data* data);
  const LazyExecutionMode& getExecutionMode() {
    auto mode = per_thread_execution_mode.find(pthread_self());
    if (mode != std::end(per_thread_execution_mode)) {
      return mode->second;
    }
    // If its the first time we are calling it for the thread it means its not
    // initialized yet and we can mark it in lazy mode as threads start from
    // there Need to check if threads can start executing from lowering statge
    // itself?
    per_thread_execution_mode[pthread_self()] = kLAZY;
    return per_thread_execution_mode[pthread_self()];
  }
  void setExecutionMode(LazyExecutionMode mode) {
    per_thread_execution_mode[pthread_self()] = mode;
  }
  void MarkTensorsExecuted() {
    for (auto& tensor : m_tensor_execution_status) {
      // Mark all the tensors in executing state as done
      if (tensor.second == kEXECUTING)
        tensor.second = kEXECUTION_COMPLETE;
    }
  }
  void MarkTensorExecuting(int tensor_id) {
    auto exec_status = m_tensor_execution_status.find(tensor_id);
    if (exec_status != std::end(m_tensor_execution_status)) {
      if (exec_status->second != kEXECUTION_COMPLETE) {
        exec_status->second = kEXECUTING;
      }
    } else {
      TORCH_CHECK(
          false,
          "Habana Lazy execution : trying to set execution stage of unregistered tensor");
    }
  }
  void MarkTensorExecuted(int tensor_id) {
    auto exec_status = m_tensor_execution_status.find(tensor_id);
    if (exec_status != std::end(m_tensor_execution_status)) {
      exec_status->second = kEXECUTION_COMPLETE;
    } else {
      TORCH_CHECK(
          false,
          "Habana Lazy execution : trying to set execution stage of unregistered tensor");
    }
  }

  void MarkTensorRegistered(int tensor_id) {
    auto exec_status = m_tensor_execution_status.find(tensor_id);
    if (exec_status != std::end(m_tensor_execution_status)) {
      exec_status->second = kREGISTERED;
    } else {
      TORCH_CHECK(
          false,
          "Habana Lazy execution : trying to set execution stage of unregistered tensor");
    }
  }
  LazyTensorExecutionStatus getTensorExecutionStatus(int index) {
    auto exec_status = m_tensor_execution_status.find(index);
    if (exec_status != std::end(m_tensor_execution_status)) {
      return exec_status->second;
    } else {
      return kUN_REGISTERED;
    }
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
  // We want to retain some tensors for special cases where PT releases them
  // but because we are in lazy mode we actually need them for processing later
  // This should only be used in special cases and released on exit cleanly
  std::vector<at::Tensor> m_retained_tensor_list;

 private:
  // A map between unique lazy tensor ID and execution status
  // Although our execution modes are per thread but tensor status is per device
  // This is because we might be juggling between various threads and we want a
  // common state set for our tensors that are flowing throught the device.
  // Device view seems to be most suited for that
  std::unordered_map<int64_t, LazyTensorExecutionStatus>
      m_tensor_execution_status;
  // LazyExecutionMode : per thread execution mode is maintained
  std::unordered_map<pthread_t, LazyExecutionMode> per_thread_execution_mode;
};

class HbExecutionContextArena {
 public:
  static HbExecutionContextArena Get();
  HbExecutionContext* getDeviceExecutionContext(int device);
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
