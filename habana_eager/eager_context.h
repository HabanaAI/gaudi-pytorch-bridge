/*******************************************************************************
 * Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
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

#include <future>
#include <mutex>
#include "backend/habana_device/HPUDevice.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"

namespace habana {
namespace eager {

/**
 * Class to store the eager context which we might need across the Ops.
 * This might have members/functionalities related to pipelining as well
 * other eager development/feature.
 */
class SingleTonEagerContext {
 public:
  /**
   * Obtains instance of context.
   * Thread safe.
   */
  static SingleTonEagerContext& getInstance() {
    std::call_once(initialize_once_flag_, CreateInstance);
    return *instance_;
  }

  /**
   * Schedule work and update handle to last scheduled work.
   * Thread safe.
   *
   * @param starter Function that launch execution and returns handle
   */
  template <class F, class... Args>
  void ScheduleWorkAndUpdateLoweringThreadHandle(F&& f, Args&&... args) {
    hpu_registrar().get_device().get_lowering_thread().enqueue<F, Args...>(
        std::forward<F>(f), std::forward<Args>(args)...);
  }

  /**
   * Joins scheduled work.
   * Ensures handle is properly obtained without data races, thus thread safe.
   */
  void JoinPendingLoweringThread();

 private:
  SingleTonEagerContext() = default;
  SingleTonEagerContext(const SingleTonEagerContext&) = delete;
  SingleTonEagerContext& operator=(const SingleTonEagerContext&) = delete;

  static std::once_flag initialize_once_flag_;
  static std::unique_ptr<SingleTonEagerContext> instance_;
  static void CreateInstance();
};

extern "C" void JoinPendingPipelineThreads();
extern "C" void JoinPendingPipelineAllThreads();
extern "C" void RestoreToOrgSendTensors(
    std::vector<at::Tensor>& tensors,
    std::vector<at::Tensor>& org_tensors);

} // namespace eager
} // namespace habana
