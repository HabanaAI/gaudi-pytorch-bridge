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

#include "backend/habana_device/HPUStream.h"

struct THP_HPU_Stream : THPStream {
  c10::hpu::HPUStream hpu_stream;
};
extern PyObject* THP_HPU_StreamClass;

void THP_HPU_Stream_init(PyObject* module);

inline bool THP_HPU_Stream_Check(PyObject* obj) {
  return THP_HPU_StreamClass && PyObject_IsInstance(obj, THP_HPU_StreamClass);
}
