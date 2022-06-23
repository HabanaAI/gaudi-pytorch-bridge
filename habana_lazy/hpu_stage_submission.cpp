/******************************************************************************
 * Copyright (C) 2021 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include "habana_lazy/hpu_stage_submission.h"
#include "lazy_executor.h"

namespace habana_lazy {

static std::atomic<size_t> compound_op_counter = 0;

PTOpTrace::PTOpTrace() {
  if (!habana_lazy::isDeviceInLoweringMode()) {
    compound_op_counter++;
  }
}

PTOpTrace::~PTOpTrace() {
  if (!habana_lazy::isDeviceInLoweringMode()) {
    compound_op_counter--;

    if (compound_op_counter == 0) {
      increment_compound_ops();
    }
  }
}

void PTOpTrace::increment_compound_ops() {
  habana_lazy::StageSubmission::getInstance().incrementCompoundOps();

  if (habana_lazy::StageSubmission::getInstance().isExceededMaxCompoundSize()) {
    PT_LAZY_DEBUG("Reached max accumulated graph size, triggering a mark_step");
    bool async = GET_ENV_FLAG_NEW(PT_HPU_ENABLE_EXECUTION_THREAD);
    habana_lazy::HbLazyTensor::StepMarker({}, nullptr, {}, async);
  }
}
} // namespace habana_lazy
