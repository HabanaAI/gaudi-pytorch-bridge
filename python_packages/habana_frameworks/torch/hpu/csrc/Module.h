/*******************************************************************************
 * Copyright (C) 2022-2024 Habana Labs, Ltd. an Intel Company
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