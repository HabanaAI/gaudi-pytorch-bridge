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

#include <torch/csrc/api/include/torch/version.h>
#include <torch/csrc/python_headers.h>

#include "backend/habana_device/HPUEvent.h"

#include <torch/csrc/Event.h>
struct THP_HPU_Event : THPEvent {
  at::hpu::HPUEvent hpu_event;
};
extern PyObject* THP_HPU_EventClass;

void THP_HPU_Event_init(PyObject* module);

inline bool THP_HPU_Event_Check(PyObject* obj) {
  return THP_HPU_EventClass && PyObject_IsInstance(obj, THP_HPU_EventClass);
}
