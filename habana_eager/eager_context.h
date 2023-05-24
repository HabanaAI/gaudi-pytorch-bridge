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

namespace habana {
namespace eager {

/*
 * Class to store the eager context which we might need across the Ops.
 * This might have members/functionalities related to pipelining as well
 * other eager development/feature.
 */

class SingleTonEagerContext {
 public:
  static SingleTonEagerContext& getInstance() {
    std::call_once(initialize_once_flag_, CreateInstance);
    return *instance_;
  }

  void JoinPendingLoweringThread();
  void StoreLoweringThreadException(std::exception_ptr exception) {
    m_lowering_thread_exception = std::move(exception);
  }
  void HandleException();
  std::future<void> m_lowering_thread_handle;

 private:
  SingleTonEagerContext() = default;
  SingleTonEagerContext(const SingleTonEagerContext&) = delete;
  SingleTonEagerContext& operator=(const SingleTonEagerContext&) = delete;
  std::exception_ptr m_lowering_thread_exception = nullptr;

  static std::once_flag initialize_once_flag_;
  static std::unique_ptr<SingleTonEagerContext> instance_;
  static void CreateInstance();
};

} // namespace eager
} // namespace habana
