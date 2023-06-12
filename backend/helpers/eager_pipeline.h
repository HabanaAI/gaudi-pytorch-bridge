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
#include "pytorch_helpers/habana_helpers/thread_pool/thread_pool.h"
#include "pytorch_helpers/habana_helpers/thread_queue.h"

namespace habana_helpers {

// singleton containing threadpool with only 1 thread. this represents one
// pipeline stage which is 'lowering' here. for any new pipeline stage we
// need to have separate singleton like this.
class SingleTonLoweringThreadPool {
 public:
  static habana_helpers::ThreadPool& getInstance() {
    static habana_helpers::ThreadPool thread_pool_obj(num_threads, QT_LockFree);
    return thread_pool_obj;
  }

 private:
  static constexpr size_t num_threads = 1;
  SingleTonLoweringThreadPool() = default;
  SingleTonLoweringThreadPool(const SingleTonLoweringThreadPool&) = delete;
  SingleTonLoweringThreadPool& operator=(const SingleTonLoweringThreadPool&) =
      delete;
};

} // namespace habana_helpers
