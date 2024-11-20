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

#include <pybind11/pybind11.h>
#include "pytorch_helpers/habana_helpers/pt_version_check.h"

PyMethodDef* THP_HPU_Module_methods();

#if IS_PYTORCH_OLDER_THAN(2, 3)
inline c10::DeviceIndex THPUtils_unpackDeviceIndex(PyObject* obj) {
  int overflow = 0;
  long value = PyLong_AsLongAndOverflow(obj, &overflow);
  if (value == -1 && PyErr_Occurred()) {
    throw python_error();
  }
  if (overflow != 0) {
    throw std::runtime_error("Overflow when unpacking DeviceIndex");
  }
  if (value > std::numeric_limits<c10::DeviceIndex>::max() ||
      value < std::numeric_limits<c10::DeviceIndex>::min()) {
    throw std::runtime_error("Overflow when unpacking DeviceIndex");
  }
  return (c10::DeviceIndex)value;
}

inline PyObject* THPUtils_packDeviceIndex(c10::DeviceIndex value) {
  return PyLong_FromLong(value);
}
#endif