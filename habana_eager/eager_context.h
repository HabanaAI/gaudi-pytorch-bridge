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

#include "backend/habana_device/HPUDevice.h"

namespace habana::eager {

template <class F, class... Args>
void ScheduleWorkAndUpdateLoweringThreadHandle(F&& f, Args&&... args) {
  HPUDeviceContext::lowering_thread().enqueue<F, Args...>(
      std::forward<F>(f), std::forward<Args>(args)...);
}

extern "C" void JoinPendingPipelineThreads();
extern "C" void JoinPendingPipelineAllThreads();
extern "C" void RestoreToOrgSendTensors(
    std::vector<at::Tensor>& tensors,
    std::vector<at::Tensor>& org_tensors);

} // namespace habana::eager
