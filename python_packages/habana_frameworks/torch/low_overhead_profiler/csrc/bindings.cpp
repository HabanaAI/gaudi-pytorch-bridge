/*******************************************************************************
 * Copyright (C) 2023-2024 Habana Labs, Ltd. an Intel Company
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
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/env_flags.h"
#include "pybind11/stl.h"
#include "pytorch_helpers/low_overhead_profiler/profiler.h"
#include "synapse_api.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "This module registers low overhead host profiler API";

  m.def("_start_lo_host_profiler", []() {
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_METRICS_COLLECTION, true, 1);
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION, true, 1);
  });
  m.def("_stop_lo_host_profiler", []() {
    habana::HPUDeviceContext::synchronize_host_multistage_pipeline();
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_METRICS_COLLECTION, false, 1);
    SET_ENV_FLAG_NEW(PT_HPU_ENABLE_LOP_TRACES_COLLECTION, false, 1);
  });
  m.def("_flush_lo_host_profiler", []() {
    LOP::ProfilerEngine::get_inst().flush();
  });
}