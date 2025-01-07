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
