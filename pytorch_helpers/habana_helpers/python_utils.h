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

#include "Python.h"

namespace habana_helpers {
// RAII structs to acquire/release Python's global interpreter lock (GIL)
// Releases the GIL on construction, if this thread already have the GIL
// acquired
struct AutoNoGIL {
  AutoNoGIL() {
    if (PyGILState_Check() && PyGILState_GetThisThreadState()) {
      save_state = PyEval_SaveThread();
    }
  }
  void Acquire() {
    if (save_state) {
      PyEval_RestoreThread(save_state);
    }
    save_state = nullptr;
  }
  ~AutoNoGIL() {
    Acquire();
  }
  PyThreadState* save_state = nullptr;
};
} // namespace habana_helpers
