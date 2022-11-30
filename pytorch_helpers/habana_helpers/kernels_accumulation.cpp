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
#include "habana_lazy/lazy_executor.h"
#include "pytorch_helpers/habana_helpers/logging.h"

#include <string>
#include <unordered_set>

namespace habana_lazy {

// list of manual ops that support parallel accumulation
const std::unordered_set<std::string> SupportedNonAutogenOps = {
    "_masked_scale",
    "_reshape_alias",
    "_unsafe_view",
    "add_",
    "add",
    "alias",
    "any",
    "as_strided_",
    "as_strided",
    "binary_cross_entropy_with_logits",
    "cat_out",
    "cat",
    "clone",
    "constant_pad_nd",
    "div_",
    "div_out",
    "div",
    "embedding_bag_sum_bwd_out",
    "embedding_bag_sum_fwd",
    "embedding_bag_sum",
    "embedding_dense_backward",
    "embedding",
    "empty_strided",
    "expand",
    "fused_norm",
    "gelu_backward",
    "gelu",
    "index_add_out",
    "index_copy_",
    "index_fill_",
    "kl_div_backward",
    "kl_div",
    "masked_fill_",
    "matmul",
    "max",
    "min",
    "one_hot",
    "permute",
    "repeat",
    "scatter_add_",
    "select",
    "slice",
    "split_with_sizes",
    "split",
    "squeeze",
    "t",
    "transpose",
    "unsqueeze_",
    "unsqueeze",
    "view"};

static std::queue<std::function<void()>> cleanup_tasks;
static std::mutex cleanup_mutex;

AccThreadPool& GetAccThreadPool() {
  static AccThreadPool thread_pool; // single thread only
  return thread_pool;
}

void PushCleanupTask(std::function<void()>&& task) {
  std::unique_lock<std::mutex> lock(cleanup_mutex);
  cleanup_tasks.emplace(std::move(task));
}

void ExecuteAllCleanupTasks() {
  PT_LAZY_TRACE

  std::queue<AccThreadPool::AccTask> empty;
  {
    std::unique_lock<std::mutex> lock(cleanup_mutex);
    // let's assume for now, that bodies of cleanup funcs are empty
    cleanup_tasks.swap(empty);
  }
}

bool IsAccThreadEnabled() {
  return GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) != 0 &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1; // only default lazy
}

bool CanUseAccThread() {
  return IsAccThreadEnabled() && !GetAccThreadPool().inThreadPool() &&
      !(SingleTonExecThreadPool::getInstance().inThreadPool() ||
        habana_lazy_executor.getDeviceExecutionContext(0)
            ->m_launch_thread_context);
}

void SyncAccThreadPool() {
  if (CanUseAccThread()) { // avoid syncing from acc and launch thread pools
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

} // namespace habana_lazy
