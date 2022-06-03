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
#include "habana_kernels/fallback_helper.h"
#include "habana_lazy/hlexec.h"
#include "habana_lazy/hpu_lazy_tensors.h"
#include "pytorch_helpers/habana_device/HPUAllocator.h"
#include "pytorch_helpers/habana_device/HPUGuardImpl.h"
#include "pytorch_helpers/habana_device/HPUStream.h"
#include "pytorch_helpers/synapse_helpers/stream.h"

int GetDeviceType() {
  auto& device = synapse_helpers::HPURegistrar::get_device();
  return device.type();
}

intptr_t GetDataPtr(const at::Tensor& t) {
  void* data_ptr;
  if (GET_ENV_FLAG_NEW(PT_HPU_LAZY_MODE) != 0) {
    data_ptr = habana_lazy::HbLazyTensor::lazyTensorDataPtr(t);
  } else {
    data_ptr = reinterpret_cast<void*>(t.storage().data_ptr().get());
  }
  size_t device_id = t.device().index();
  auto& device = synapse_helpers::HPURegistrar::get_device(device_id);

  auto address = reinterpret_cast<void*>(device.get_fixed_address(data_ptr));
  return reinterpret_cast<intptr_t>(address);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("get_device_type", []() { return GetDeviceType(); });
  m.def(
      "data_ptr",
      [](const at::Tensor& t) { return GetDataPtr(t); },
      py::arg("t"));
  m.def("compute_stream", []() {
    auto& d = synapse_helpers::HPURegistrar::get_device();
    HPUStream hpu_stream = getDefaultHPUStream(d.id());
    void* stream = (void*)d.get_compute_stream(hpu_stream.id());
    return reinterpret_cast<intptr_t>(stream);
  });
  py::enum_<synDeviceType>(m, "synDeviceType")
      .value("synDeviceGaudi", synDeviceGaudi)
      .value("synDeviceGaudiM", synDeviceGaudiM)
      .value("synDeviceGaudi2", synDeviceGaudi2)
      .export_values();

  m.doc() =
      "This module registers hpu experimental API used by Media internal component.";
}
