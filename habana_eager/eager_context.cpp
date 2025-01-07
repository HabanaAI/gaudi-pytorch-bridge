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

#include "eager_context.h"
#include "backend/habana_device/hpu_cached_devices.h"

namespace habana::eager {

void JoinPendingPipelineThreads() {
  HPUDeviceContext::join_pipeline_threads();
}

void JoinPendingPipelineAllThreads() {
  HPUDeviceContext::join_all_threads();
}

// Restore tensors to the org tensors for eager send P2P collective
void RestoreToOrgSendTensors(
    std::vector<at::Tensor>& tensors,
    std::vector<at::Tensor>& org_tensors) {
  HABANA_ASSERT(
      tensors.size() == org_tensors.size(),
      "Eager send tensors size not equal to org tensors");
  for (size_t i = 0; i < tensors.size(); i++) {
    tensors[i] = org_tensors[i];
  }
}

} // namespace habana::eager
