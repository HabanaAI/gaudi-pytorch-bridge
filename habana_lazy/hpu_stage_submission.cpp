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
#include "habana_lazy/hpu_stage_submission.h"
#include "lazy_executor.h"
#include "pytorch_helpers/habana_helpers/kernels_accumulation.h"

namespace habana_lazy {

static std::atomic<size_t> compound_op_counter = 0;

PTOpTrace::PTOpTrace() {
  if (habana_lazy::AccThread::IsAccThreadEnabled() &&
      habana_lazy::AccThread::Get().inAccThreadContext()) {
    // Avoid incrementing/decrementing compund op counter in acc thread. This
    // can cause counting of internal ops. Further this ensures that StepMarker
    // is not called from the accumulation thread. StepMarker can deallocate
    // tensors and it can cause deadlock.
    return;
  }

  if (!habana_lazy::isDeviceInLoweringMode()) {
    compound_op_counter++;
  }
}

PTOpTrace::~PTOpTrace() {
  if (habana_lazy::AccThread::IsAccThreadEnabled() &&
      habana_lazy::AccThread::Get().inAccThreadContext()) {
    // Avoid incrementing/decrementing compund op counter in acc thread. This
    // can cause counting of internal ops. Further this ensures that StepMarker
    // is not called from the accumulation thread. StepMarker can deallocate
    // tensors and it can cause deadlock.
    return;
  }

  if (!habana_lazy::isDeviceInLoweringMode()) {
    compound_op_counter--;

    if (compound_op_counter == 0) {
      increment_compound_ops();
    }
  }
}

void PTOpTrace::increment_compound_ops() {
  habana_lazy::StageSubmission::getInstance().incrementCompoundOps();

  if ((habana_lazy::StageSubmission::getInstance()
           .isExceededMaxCompoundSize())) {
    PT_LAZY_DEBUG(
        "Reached max accumulated compound op size, triggering a mark_step");
    bool async = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD) and
        not GET_ENV_FLAG_NEW(PT_HPU_MAX_COMPOUND_OP_SYNC);
    PT_IRGRAPH_DEBUG(
        "step marker due to reaching max accumulated compound op size");
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, async);
  }
}
} // namespace habana_lazy
