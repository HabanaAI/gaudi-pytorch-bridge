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

#include <torch/csrc/python_headers.h>

#include "backend/habana_device/HPUEvent.h"

struct THP_HPU_Event {
  PyObject_HEAD at::hpu::HPUEvent hpu_event;
};
extern PyObject* THP_HPU_EventClass;

void THP_HPU_Event_init(PyObject* module);

inline bool THP_HPU_Event_Check(PyObject* obj) {
  return THP_HPU_EventClass && PyObject_IsInstance(obj, THP_HPU_EventClass);
}
