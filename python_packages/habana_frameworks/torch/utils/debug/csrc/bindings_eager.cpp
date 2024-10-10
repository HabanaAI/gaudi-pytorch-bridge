/*******************************************************************************
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
#include <torch/extension.h>
#include "backend/kernel/hpu_habana_launch_op_pt.h"
#include "habana_eager/eager_context.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("bridge_cleanup", []() {
    try {
      habana::eager::JoinPendingPipelineThreads();
    } catch (const c10::Error& e) {
    }
    habana::HabanaLaunchOpPT::cleanUp();
  });
  m.def("join_pending_pipeline_threads", []() {
    habana::eager::JoinPendingPipelineThreads();
  });
}
