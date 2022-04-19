/******************************************************************************
 * Copyright (C) 2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 * Unauthorized copying of this file, via any medium is strictly prohibited.
 * Proprietary and confidential.
 *
 ******************************************************************************
 */
#include <pybind11/chrono.h>
#include <synapse_common_types.h>
#include <torch/extension.h>
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/synapse_helpers/stream.h"

void hpu_init() {
  habana::HABANAGuardImpl device_guard;
  device_guard.getDevice();
  auto& device = synapse_helpers::HPURegistrar::get_device();
  device.get_count_by_current_type();
  // later will add device properties here.
}

const std::string get_device_name(int device_id) {
  // We don't support index addresed device and for multi node
  // runs, every node has seperate copy of synapse lib and will
  // get device with index 0, so ignoring device_id for now.
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.name();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", []() { hpu_init(); });
  m.def("current_device", []() {
    auto& d = synapse_helpers::HPURegistrar::get_device();
    return d.id();
  });
  m.def("synchronize_device", []() {
    synapse_helpers::HPURegistrar::synchronize_device();
  });
  m.def("device_count", []() {
    return synapse_helpers::HPURegistrar::get_total_device_count();
  });
  m.def("get_device_name", [](int id) { return get_device_name(id); });
  m.doc() = "This module registers hpu backend.";
}
