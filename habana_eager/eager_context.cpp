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

#include "eager_context.h"
#include <c10/macros/Macros.h>
#include <future>
#include <memory>
#include <mutex>
#include "backend/habana_device/HPUStream.h"
#include "backend/habana_device/hpu_cached_devices.h"
#include "backend/helpers/eager_pipeline.h"
#include "pytorch_helpers/habana_helpers/logging.h"
#include "pytorch_helpers/habana_helpers/python_utils.h"
#include "pytorch_helpers/habana_helpers/thread_queue.h"

namespace habana {
namespace eager {

std::once_flag SingleTonEagerContext::initialize_once_flag_{};
std::unique_ptr<SingleTonEagerContext> SingleTonEagerContext::instance_{
    nullptr};

void SingleTonEagerContext::CreateInstance() {
  instance_.reset(new SingleTonEagerContext());
  habana::hpu_registrar().register_eager_context(
      []() { instance_.reset(nullptr); });
  c10::hpu::setJoinEagerThreadsCB(habana::eager::JoinPendingPipelineAllThreads);
}

void SingleTonEagerContext::JoinPendingLoweringThread() {
  PT_EAGER_TRACE;
  if (!hpu_registrar().is_initialized())
    return;

  try {
    // TODO remove gil_release once SW-160978 is fixed
    habana_helpers::AutoNoGIL gil_release;
    hpu_registrar().get_device().get_lowering_thread().waitWorkComplete();
  } catch (const std::exception& e) {
    hpu_registrar().get_device().set_exception_occurred(true);
    PT_BRIDGE_FATAL("Exception in Lowering thread...\n", e.what());
  } catch (...) {
    hpu_registrar().get_device().set_exception_occurred(true);
    PT_BRIDGE_FATAL("Exception in Lowering thread...\n");
  }
}

void JoinPendingPipelineThreads() {
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();
  habana_helpers::Singleton_CompileThreadPool::getInstance()
      .JoinPendingThread();
  habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
}

void JoinPendingPipelineAllThreads() {
  habana::eager::SingleTonEagerContext::getInstance()
      .JoinPendingLoweringThread();
  habana_helpers::Singleton_CompileThreadPool::getInstance()
      .JoinPendingThread();
  habana_helpers::Singleton_ExecThreadPool::getInstance().JoinPendingThread();
  habana_helpers::Singleton_GarbageCollectionThreadPool::getInstance()
      .JoinPendingThread();
}

// Method for restoring odd size tensors for eager collectives
void RestoreOddSizeSendTensors(std::vector<at::Tensor>& tensors) {
  // Join pending is required since it is called in the main thread.
  // But this should not affect the performance as it is called for
  // corner case of odd size tensors, which it self calls resize
  // which calls Join pending since src tensor is in the pipeline.
  // If no permutation, restore to the original tensor.
  JoinPendingPipelineThreads();
  for (auto& tensor : tensors) {
    auto tensor_tmeta{habana::get_tensor_extra_meta(tensor)};
    if (auto org_tensor = tensor_tmeta->get_send_org_tensor()) {
      tensor = *org_tensor;
    }
  }
}

} // namespace eager
} // namespace habana
