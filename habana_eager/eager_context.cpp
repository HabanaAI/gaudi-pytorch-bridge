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
