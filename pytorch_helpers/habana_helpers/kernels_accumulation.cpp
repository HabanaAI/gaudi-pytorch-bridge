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

#include <string>
#include <unordered_set>

namespace habana_lazy {

// list of manual ops that support parallel accumulation
const std::unordered_set<std::string> SupportedNonAutogenOps = {
    "_masked_scale",
    "add",
    "add_",
    "any",
    "binary_cross_entropy_with_logits",
    "cat",
    "cat_out",
    "clone",
    "constant_pad_nd",
    "div",
    "div_",
    "div_out",
    "embedding",
    "embedding_bag_sum",
    "embedding_bag_sum_bwd_out",
    "embedding_bag_sum_fwd",
    "embedding_dense_backward",
    "empty_strided",
    "gelu",
    "gelu_backward",
    "fused_norm",
    "kl_div",
    "kl_div_backward",
    "matmul",
    "max",
    "min",
    "one_hot",
    "masked_fill_",
    "repeat",
    "scatter_add_"};
// black list of aut-gen ops that do not support parallel accumulation
const std::unordered_set<std::string> AccThreadOpsBlacklist = {};

static std::queue<AccThreadPool::AccTask> cleanup_tasks;
static std::mutex cleanup_mutex;

AccThreadPool& GetAccThreadPool() {
  static AccThreadPool thread_pool; // single thread only
  return thread_pool;
}

void PushCleanupTask(AccThreadPool::AccTask&& task) {
  std::unique_lock<std::mutex> lock(cleanup_mutex);
  cleanup_tasks.emplace(std::move(task));
}

void ExecuteAllCleanupTasks() {
  PT_LAZY_TRACE

  std::unique_lock<std::mutex> lock(cleanup_mutex);
  while (!cleanup_tasks.empty()) {
    cleanup_tasks
        .pop(); // let's assume for now, that bodies of cleanup funcs are empty
  }
}

bool IsAccThreadEnabled() {
  return GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) != 0;
}

bool CanUseAccThread() {
  return IsAccThreadEnabled() && !GetAccThreadPool().inThreadPool();
}

void SyncAccThreadPool() {
  if (CanUseAccThread()) { // avoid syncing from within thread pool
    PT_LAZY_TRACE
    PT_LAZY_PARALLEL_ACC_DEBUG("Synchronizing accumulation thread ...");
    GetAccThreadPool().waitWorkComplete();
    ExecuteAllCleanupTasks();
  }
}

void SyncManualOpIfNeeded(const std::string& op) {
  if (IsAccThreadEnabled()) {
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
