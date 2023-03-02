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
    static SingleTonEagerContext eager_context_obj;
    return eager_context_obj;
  }

  void JoinPendingLoweringThread();
  void HandleException();
  std::future<void> m_lowering_thread_handle;

 private:
  SingleTonEagerContext() = default;
  SingleTonEagerContext(const SingleTonEagerContext&) = delete;
  SingleTonEagerContext& operator=(const SingleTonEagerContext&) = delete;
  std::exception_ptr m_lowering_thread_exception_handler = nullptr;
};

} // namespace eager
} // namespace habana
