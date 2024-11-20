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

#include <string>
#include <unordered_set>
#include "pytorch_helpers/habana_helpers/thread_pool/acc_thread_pool.h"

namespace habana_lazy {

class AccThread {
 public:
  // returns main accumulation thread pool
  static AccThread& Get() {
    std::call_once(initialize_once_flag_, CreateInstance);
    HABANA_ASSERT(instance_);
    return *instance_;
  }

  bool inAccThreadContext() const;
  void run(std::function<void()>&& func);
  void discardPendingTasks();

  // store cleanup tasks that are holding any resources used by accumulation
  // thread pool Its purpose is to avoid deadlock on GIL.
  //
  // Deadlock in GIL happens when main thread is calling C++ code from Python
  // (that acquires GIL by default) and accumulation thread is finishing a
  // previous task, which at the end can release Python resources - in this case
  // at::Tensors. Main thread is trying to synchronize accumulation thread (due
  // to unsupported op) and accumulation thread is trying to acquire GIL to
  // release resources. It is avoided by moving any resource from accumulation
  // to cleanup thread. Cleanup thread is not synchronized by main thread and
  // can safely wait for GIL release.
  void PushCleanupTask(std::function<void()>&& task);

  // In order to avoid non-deterministic order of resource deallocation, this is
  // called only in one place in StepMarker together with AccThread
  // synchronization before execution.
  void ExecuteAllCleanupTasks();

  // checks if accumulation thread is enabled
  static bool IsAccThreadEnabled();
  // checks if accumulation thread can be used
  bool CanUseAccThread();
  // synchronizes acc thread pool, if parallel accumulation is enabled
  void SyncAccThreadPool();
  // synchronizes acc thread pool if the input manual 'op' is not supported
  // for parallel accumulation.
  //
  // Used only for manual ops, not auto-gen.
  void SyncManualOpIfNeeded(const std::string& op);

  static thread_local bool acc_thread_allowed;

 private:
  AccThread();
  std::queue<std::function<void()>> cleanup_tasks;
  std::mutex cleanup_mutex;
  static const std::unordered_set<std::string> SupportedNonAutogenOps;
  static std::once_flag initialize_once_flag_;
  static std::unique_ptr<AccThread> instance_;
  std::unique_ptr<AccThreadPoolBase> thread_pool;

  bool CanUseAccThreadInternal();
  static void CreateInstance();
};

// Class to manage global state to disable the accumulation thread in some
// context i.e. complex ops mixing acc enabled ops with non-enabled. Once all
// ops are moved to the acc thread infrastructure, this class can be removed.
class NoAccThread {
 public:
  NoAccThread(bool sync_acc_thread = true);
  NoAccThread(const NoAccThread&) = delete;
  NoAccThread(NoAccThread&&) = delete;
  ~NoAccThread();

 private:
  // set to true when state of accumulation thread was updated
  bool update_state_ = false;
};

} // namespace habana_lazy
