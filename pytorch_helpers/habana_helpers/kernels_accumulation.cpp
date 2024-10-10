/******************************************************************************
 * Copyright (C) 2022-2023 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/hpu_cached_devices.h"
#include "habana_helpers/logging.h"
#include "habana_lazy/lazy_executor.h"

#include <memory>
#include <string>
#include <unordered_set>

namespace habana_lazy {

// list of manual ops that support parallel accumulation
const std::unordered_set<std::string> AccThread::SupportedNonAutogenOps = {
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
    "fused_clip_norm",
    "gelu_backward",
    "gelu",
    "index_add_out",
    "index_fill_",
    "kl_div_backward",
    "kl_div",
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

thread_local bool AccThread::acc_thread_allowed = true;
std::unique_ptr<AccThread> AccThread::instance_{nullptr};
std::once_flag AccThread::initialize_once_flag_{};

void AccThread::CreateInstance() {
  instance_.reset(new AccThread());
  habana::hpu_registrar().register_acc_thread(
      []() { instance_.reset(nullptr); });
}

AccThread::AccThread() : thread_pool(CreateAccThreadPool()) {}

bool AccThread::inAccThreadContext() const {
  return thread_pool->inAccThreadContext();
}

void AccThread::run(std::function<void()>&& func) {
  thread_pool->run(std::move(func));
}

void AccThread::discardPendingTasks() {
  thread_pool->discardPendingTasks();
}

void AccThread::PushCleanupTask(std::function<void()>&& task) {
  std::unique_lock<std::mutex> lock(cleanup_mutex);
  cleanup_tasks.emplace(std::move(task));
}

bool AccThread::IsAccThreadEnabled() {
  return GET_ENV_FLAG_NEW(PT_HPU_LAZY_ACC_PAR_MODE) != 0 &&
      GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) == 1; // only default lazy
}

bool AccThread::CanUseAccThreadInternal() {
  return IsAccThreadEnabled() &&
      !inAccThreadContext(); // avoid using acc thread pool inside the pool
}

bool AccThread::CanUseAccThread() {
  return acc_thread_allowed && CanUseAccThreadInternal();
}

void AccThread::ExecuteAllCleanupTasks() {
  if (!CanUseAccThreadInternal()) {
    return;
  }

  PT_LAZY_TRACE
  std::queue<AccThreadPoolBase::AccTask> empty;
  {
    std::unique_lock<std::mutex> lock(cleanup_mutex);
    // let's assume for now, that bodies of cleanup funcs are empty
    cleanup_tasks.swap(empty);
  }
}

void AccThread::SyncAccThreadPool() {
  if (CanUseAccThreadInternal()) {
    PT_LAZY_TRACE
    PT_LAZY_PARALLEL_ACC_DEBUG("Synchronizing accumulation thread ...");
    thread_pool->waitWorkComplete();
  }
}

void AccThread::SyncManualOpIfNeeded(const std::string& op) {
  if (IsAccThreadEnabled()) {
    if (!SupportedNonAutogenOps.count(op)) {
      PT_LAZY_PARALLEL_ACC_DEBUG(
          op, " op not supported for parallel accumulation");
      SyncAccThreadPool();
    }
  }
}

NoAccThread::NoAccThread(bool sync_acc_thread) {
  update_state_ = AccThread::Get().CanUseAccThread();
  if (update_state_) {
    if (sync_acc_thread) {
      AccThread::Get().SyncAccThreadPool();
    }
    AccThread::acc_thread_allowed = false;
  }
}

NoAccThread::~NoAccThread() {
  if (update_state_) {
    AccThread::acc_thread_allowed = true;
  }
}

} // namespace habana_lazy
