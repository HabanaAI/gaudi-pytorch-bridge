/*******************************************************************************
 * Copyright (C) 2020-2023 Habana Labs, Ltd. an Intel Company
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
#include <pybind11/chrono.h>
#include <torch/extension.h>
#include "backend/kernel/hpu_habana_cache.h"
#include "habana_kernels/fallback_helper.h"
#include "habana_kernels/random_gen_kernels.h"
#include "habana_lazy/hlexec.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"

namespace {
int GetCurrentThreadDevice() {
  auto& d = synapse_helpers::HPURegistrar::get_device();
  return d.id();
}
} // namespace

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
  m.def("_get_default_generator", []() {
    return habana::getDefaultHPUGenerator();
  });
  m.doc() = "This module registers hpu lazy api.";
}
