/******************************************************************************
 * Copyright (C) 2020 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <pybind11/chrono.h>
#include <torch/extension.h>
#include "habana_bridge/kernel/hpu_habana_cache.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"

int GetCurrentThreadDevice() {
  auto& d = synapse_helpers::HPURegistrar::get_device();
  return d.id();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("_hb_get_default_device", []() { return GetCurrentThreadDevice(); });
  m.def(
      "_iter_mark_step", []() { habana_lazy::HbLazyTensor::IterStepMarker(); });
  m.def(
      "_mark_step",
      [](const std::string& device_str) {
        habana_lazy::HbLazyTensor::StepMarkerBind(device_str);
      },
      py::arg("device_str") = "");

  m.doc() = "This module registers hpu lazy api.";
}