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
