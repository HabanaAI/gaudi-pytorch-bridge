/******************************************************************************
 * Copyright (C) 2022 Habana Labs, Ltd. an Intel Company
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
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"

#include "pytorch_helpers/habana_helpers/logging.h"

#include <ATen/PTThreadPool.h>
#include <string>
#include <unordered_set>

namespace habana_lazy {

// list of manual ops that support parallel accumulation
const std::unordered_set<std::string> SupportedNonAutogenOps = {"div"};
// black list of aut-gen ops that do not support parallel accumulation
const std::unordered_set<std::string> AccThreadOpsBlacklist = {
    "mul" // due to using hpu::mul() in mul_out_hpu_lazy(), there is deadlock on
          // recursive lock used for view handling
};

at::PTThreadPool& GetAccThreadPool() {
  static at::PTThreadPool thread_pool(1); // single thread only
  return thread_pool;
}

at::PTThreadPool& GetAccCleanupThreadPool() {
  static at::PTThreadPool thread_pool(1); // single thread only
  return thread_pool;
}

void SyncAccThreadPool() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) != 0 &&
      !GetAccThreadPool()
           .inThreadPool()) { // avoid syncing from within thread pool
    PT_LAZY_PARALLEL_ACC_DEBUG("Synchronizing accumulation thread ...");
    GetAccThreadPool().waitWorkComplete();
  }
}

void SyncCleanupThreadPool() {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) != 0 &&
      !GetAccCleanupThreadPool()
           .inThreadPool()) { // no syncing from within thread pool
    PT_LAZY_PARALLEL_ACC_DEBUG("Synchronizing accumulation cleanup thread ...");
    GetAccCleanupThreadPool().waitWorkComplete();
  }
}

void SyncManualOpIfNeeded(const std::string& op) {
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) != 0) {
    if (!SupportedNonAutogenOps.count(op)) {
      PT_LAZY_PARALLEL_ACC_DEBUG(
          op, " op not supported for parallel accumulation");
      SyncAccThreadPool();
    }
  }
}

bool IsAccumulationForAutogenSupported(const std::string& op) {
  return !AccThreadOpsBlacklist.count(op);
}

} // namespace habana_lazy
